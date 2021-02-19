'''Bla bla

'''
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

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

#
# Connect to raw data images
local_imgs = image_factory.create('local disk', folder='./data_tmp')

#
# Connect to image segmentor and execute to create smaller set of single cell images
segment_creator = CellImageSegmentor(return_channels=('green',))
n_channels_in = len(segment_creator.return_channels)
for cell_id, data_path_collection in local_imgs.items():
    #segment_creator.transform(data_path_collection).write_entry(cell_id)
    if not segment_creator.already_in_db_(cell_id):
        segment_creator.transform(data_path_collection).write_entry(cell_id)

#
# Connect to dataset over all available single cell images
cellsegment_dataset = CellImageSegmentOneClassContrastDataset(positive_one_class=14,
                                                              cell_image_segmentor=segment_creator,
                                                              data_label_file='./data_tmp/train.csv')
dloader = DataLoader(cellsegment_dataset, batch_size=2, shuffle=True)

#
# Define model and loss function
model = SupConResNet(name='resnet18', feat_dim=128, in_channel=n_channels_in)
criterion = SupConLoss()

#
# Define optimizer
optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

#
# Training loop
n_epochs = 2
for epoch in range(n_epochs):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (labels, images) in enumerate(dloader):

        print (images.shape)
        view_1, view_2 = torch.unbind(images, dim=1)
        images = torch.cat([view_1, view_2], dim=0)

        features = model(images)
        f1, f2 = torch.split(features, [2, 2], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        losses.update(loss.item(), 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print_freq = 1
        if (idx + 1) % print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(dloader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

return losses.avg