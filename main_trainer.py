'''Bla bla

'''
from trainer import TrainerImageSegmentBinaryContrastive

t14 = TrainerImageSegmentBinaryContrastive(data_batch_size=64,
                                           data_positive_minratio=0.15,
                                           save_model_path='t14_saved.tar')
t14.train_me(1,1)

