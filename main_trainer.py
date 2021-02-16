'''Bla bla

'''
from train_data import image_factory, contrast_split, parse_labels
from dataset import CellImageSegmentsTransform

local_imgs = image_factory.create('local disk', folder='./data_tmp')
df_labels = parse_labels('./data_tmp/train.csv')
df1, df2 = contrast_split(df_labels, 14)

segment_me_ = CellImageSegmentsTransform()

for cell_id, data_path_collection in local_imgs.items():
    rets = segment_me_(data_path_collection)
    print (rets)
    raise RuntimeError