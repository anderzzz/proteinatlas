'''Bla bla

'''
import torch
from torch import nn

from _learner import _Learner
from supcontrast.resnet_big import SupConResNet
from supcontrast.losses import SupConLoss

class LearnerSupCon(_Learner):

    def __init__(self,
                 resnet_name='resnet50',
                 feature_dim=128,
                 synchronized_batch_norm=True
                 ):

        super().__init__()

        self.inp_resnet_name = resnet_name
        self.inp_feature_dim = feature_dim
        self.inp_synchronized_batch_norm = synchronized_batch_norm

        self.model = SupConResNet(name=self.inp_resnet_name, feat_dim=self.inp_feature_dim).to(self.device)
        self.criterion = SupConLoss.to(self.device)

        self.set_sgd_optim()

        self.print_inp()