'''Bla bla

'''
import torch

from trainer import TrainerImageSegmentBinaryContrastive

t14 = TrainerImageSegmentBinaryContrastive(data_batch_size=32,
                                           data_positive_minratio=0.15,
                                           save_model_path='t14_saved.tar',
                                           model_data_precision=torch.float64)
t14.train_me(2,1)

