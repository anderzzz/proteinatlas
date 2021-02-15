'''Parent class for learners

'''
import sys
import time
import abc

from numpy.random import seed, randint

import torch
from torch.utils.data import DataLoader
from torch import optim

class LearnerInterface(metaclass=abc.ABCMeta):
    '''Formal interface for the Learner subclasses. Any class inheriting `_Learner` will have to satisfy this
    interface, otherwise it will not instantiate
    '''
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'eval') and
                callable(subclass.eval) and
                hasattr(subclass, 'save_model') and
                callable(subclass.save_model) and
                hasattr(subclass, 'load_model') and
                callable(subclass.load_model))

    @abc.abstractmethod
    def train(self, n_epochs: int):
        '''Train model'''
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, **kwargs):
        '''Evaluate model'''
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, path: str):
        '''Save model state to file'''
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, path: str):
        '''Save model state to file'''
        raise NotImplementedError


class _Learner(LearnerInterface):
    '''Parent class for learners on the cell image data

    '''
    def __init__(self, run_label='', random_seed=None, f_out=sys.stdout,
                 raw_csv_toc=None, raw_csv_root=None,
                 save_tmp_name='model_in_training',
                 selector=None, iselector=None,
                 dataset_type='full basic', dataset_kwargs={},
                 loader_batch_size=16, num_workers=0,
                 show_batch_progress=True, deterministic=True,
                 epoch_conclude_func=None):

        self.inp_run_label = run_label
        self.inp_random_seed = random_seed
        self.inp_f_out = f_out
        self.inp_raw_csv_toc = raw_csv_toc
        self.inp_raw_csv_root = raw_csv_root
        self.inp_save_tmp_name = save_tmp_name
        self.inp_selector = selector
        self.inp_iselector = iselector
        self.inp_dataset_type = dataset_type
        self.inp_dataset_kwargs = dataset_kwargs
        self.inp_loader_batch_size = loader_batch_size
        self.inp_num_workers = num_workers
        self.inp_show_batch_progress = show_batch_progress
        self.inp_deterministic = deterministic
        if epoch_conclude_func is None:
            self.inp_epoch_conclude_func = lambda: None
        else:
            self.inp_epoch_conclude_func = epoch_conclude_func

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        seed(self.inp_random_seed)
        torch.manual_seed(randint(2**63))
        if self.inp_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

#        self.dataset = XXX
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.inp_loader_batch_size,
                                     shuffle=True,
                                     num_workers=self.inp_num_workers)

        self.optimizer = None
        self.lr_scheduler = None

    def set_sgd_optim(self, parameters, lr=0.01, momentum=0.9, weight_decay=0.0,
                      scheduler_step_size=15, scheduler_gamma=0.1):
        '''Bla bla

        '''
        self.optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                      step_size=scheduler_step_size,
                                                      gamma=scheduler_gamma)

    def print_inp(self):
        '''Output input parameters for easy reference in future. Based on naming variable naming convention.
        '''
        the_time = time.localtime()
        print('Run at {}/{}/{} {}:{}:{} with arguments:'.format(the_time.tm_year, the_time.tm_mon, the_time.tm_mday,
                                                                the_time.tm_hour, the_time.tm_min, the_time.tm_sec),
              file=self.inp_f_out)
        for attr_name, attr_value in self.__dict__.items():
            if 'inp_' == attr_name[0:4]:
                key = attr_name[4:]
                print('{} : {}'.format(key, attr_value), file=self.inp_f_out)