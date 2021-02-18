'''Bla bla

'''
from dataset import CellImageSegmentContrastDataset

segdata = CellImageSegmentContrastDataset(img_segments_folder='./data_tmp_segments',
                                          data_label_file='./data_tmp/train.csv')

for dd in segdata:
    print (dd)
    raise RuntimeError