'''Bla bla

'''
import torch

from trainer import TrainerImageSegmentBinaryContrastive, create_segments

segment_handler = create_segments('local disk', './data_tmp', ('green',), False, 4)
t14 = TrainerImageSegmentBinaryContrastive(data_batch_size=4,
                                           data_positive_minratio=0.15,
                                           data_segment_handler=segment_handler,
                                           save_model_path='t14_saved.tar',
                                           model_data_precision=torch.float64)
t14.load_me('t14_saved.tar')
t14.train_me(1,1)

