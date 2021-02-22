'''Bla bla

'''
from trainer import TrainerImageSegmentBinaryContrastive

t14 = TrainerImageSegmentBinaryContrastive(data_batch_size=32,
                                           data_positive_minratio=None)
t14.train_me(1,1)

