'''Bla bla

'''
import sys
import time

from numpy.random import shuffle

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import SGD, lr_scheduler

import tensorboard_logger as tb_logger

from train_data import image_factory
from cellsegments import CellImageSegmentor
from dataset import CellImageSegmentOneClassContrastDataset

from supcontrast.resnet_big import SupConResNet
from supcontrast.losses import SupConLoss

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainerImageSegmentBinaryContrastive(object):
    '''Bla bla

    '''
    def __init__(self,
                 raw_image_src_type='local disk',
                 raw_image_folder='./data_tmp',
                 raw_image_channels=('green',),
                 raw_image_data_label_file='./data_tmp/train.csv',
                 raw_image_resegmentation=False,
                 data_batch_size=128,
                 data_positive_class=14,
                 data_positive_minratio=None,
                 model_data_precision=torch.float32,
                 model_name='resnet18',
                 model_feat_dim=128,
                 opt_lr=0.05,
                 opt_momentum=0.9,
                 opt_weight_decay=1e-4,
                 opt_scheduler_stepsize=15,
                 opt_scheduler_gamma=0.1,
                 save_model_path='hello',
                 save_model_freq=10,
                 save_logger_folder='./logging'
                 ):

        self.inp = {
            'raw_image_src_type' : raw_image_src_type,
            'raw_image_folder' : raw_image_folder,
            'raw_image_channels' : raw_image_channels,
            'raw_image_data_label_file' : raw_image_data_label_file,
            'raw_image_resegmentation' : raw_image_resegmentation,
            'data_batch_size' : data_batch_size,
            'data_positive_class' : data_positive_class,
            'data_positive_minratio' : data_positive_minratio,
            'model_data_precision' : model_data_precision,
            'model_name' : model_name,
            'model_feat_dim' : model_feat_dim,
            'opt_lr' : opt_lr,
            'opt_momentum' : opt_momentum,
            'opt_weight_decay' : opt_weight_decay,
            'opt_scheduler_stepsize' : opt_scheduler_stepsize,
            'opt_scheduler_gamma' : opt_scheduler_gamma,
            'save_model_path' : save_model_path,
            'save_model_freq' : save_model_freq,
            'save_logger_folder' : save_logger_folder
            }

        local_imgs = image_factory.create(self.inp['raw_image_src_type'],
                                          folder=self.inp['raw_image_folder'])
        segment_creator = CellImageSegmentor(return_channels=self.inp['raw_image_channels'])
        n_channels_in = len(segment_creator.return_channels)
        if self.inp['raw_image_resegmentation']:
            segment_creator.reset()
        for cell_id, data_path_collection in local_imgs.items():
            if not segment_creator.already_in_db_(cell_id):
                segment_creator.transform(data_path_collection).write_entry(cell_id)

        cellsegment_dataset = CellImageSegmentOneClassContrastDataset(positive_one_class=self.inp['data_positive_class'],
                                                                      cell_image_segmentor=segment_creator,
                                                                      data_label_file=self.inp['raw_image_data_label_file'],
                                                                      image_dtype=self.inp['model_data_precision'])
        if not self.inp['data_positive_minratio'] is None:
            i_pos = cellsegment_dataset.positive_items
            i_neg = cellsegment_dataset.negative_items
            if len(i_pos) / (len(i_pos) + len(i_neg)) < self.inp['data_positive_minratio']:
                n_negs = len(i_pos) / self.inp['data_positive_minratio'] - len(i_pos)
                shuffle(i_neg)
                i_neg = i_neg[:int(n_negs)]
            subsetsampler = SubsetRandomSampler(i_pos + i_neg)
            self.dloader = DataLoader(cellsegment_dataset,
                                      batch_size=self.inp['data_batch_size'],
                                      sampler=subsetsampler)
        else:
            self.dloader = DataLoader(cellsegment_dataset,
                                      batch_size=self.inp['data_batch_size'],
                                      shuffle=True)

        self.model = SupConResNet(name=self.inp['model_name'],
                                  feat_dim=self.inp['model_feat_dim'],
                                  in_channel=n_channels_in)
        self.model = self.model.type(self.inp['model_data_precision'])
        self.criterion = SupConLoss()

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model.encoder = torch.nn.DataParallel(self.model.encoder)
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.optimizer = SGD(self.model.parameters(),
                             lr=self.inp['opt_lr'],
                             momentum=self.inp['opt_momentum'],
                             weight_decay=self.inp['opt_weight_decay'])
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                step_size=self.inp['opt_scheduler_stepsize'],
                                                gamma=self.inp['opt_scheduler_gamma'])

        self.logger = tb_logger.Logger(logdir=self.inp['save_logger_folder'], flush_secs=2)

    def train_me(self, n_epochs=1, print_freq=10):
        '''Bla bla

        '''
        for epoch in range(n_epochs):
            self.model.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            end = time.time()
            for idx, (labels, images) in enumerate(self.dloader):
                data_time.update(time.time() - end)

                batch_size = labels.shape[0]

                print (images.shape)
                view_1, view_2 = torch.unbind(images, dim=1)
                images = torch.cat([view_1, view_2], dim=0)

                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                features = self.model(images)
                f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = self.criterion(features, labels)

                losses.update(loss.item(), batch_size)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                batch_time.update(time.time() - end)
                end = time.time()

                if (idx + 1) % print_freq == 0:
                    print('Train: [{0}][{1}/{2}]\t'
                          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                        epoch, idx + 1, len(self.dloader), batch_time=batch_time,
                        data_time=data_time, loss=losses))
                    sys.stdout.flush()

            self.logger.log_value('loss', losses.avg, epoch)
            self.logger.log_value('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            if epoch % self.inp['save_model_freq'] == 0:
                self.save_me(epoch)

        self.save_me(n_epochs)

    def save_me(self, epoch=-1):
        '''Bla bla

        '''
        state = {
            'opt': self.optimizer,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, self.inp['save_model_path'])
        del state

    def load_me(self, model_path):
        '''Bla bla

        '''
        saved_dict = torch.load(model_path)
        self.model.load_state_dict(saved_dict['model'])
        self.optimizer.load_state_dict(saved_dict['optimizer'])