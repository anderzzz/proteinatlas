'''Bla bla

'''
from train_data import image_factory, contrast_split, parse_labels
from cellsegments import CellImageSegmentor

local_imgs = image_factory.create('local disk', folder='./data_tmp')
df_labels = parse_labels('./data_tmp/train.csv')
df1, df2 = contrast_split(df_labels, 14)

segment_creator = CellImageSegmentor(return_channels=('green','red','blue'))
print (segment_creator)

for cell_id, data_path_collection in local_imgs.items():
    segment_creator.transform(data_path_collection).write_entry(cell_id)