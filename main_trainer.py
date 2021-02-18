'''Bla bla

'''
from torch.utils.data import DataLoader

from train_data import image_factory, contrast_split, parse_labels
from cellsegments import CellImageSegmentor
from dataset import CellImageSegmentContrastDataset, CellImageSegmentBatchSampler, CellImageSegmentRandomSampler

local_imgs = image_factory.create('local disk', folder='./data_tmp')
df_labels = parse_labels('./data_tmp/train.csv')
df1, df2 = contrast_split(df_labels, 14)

segment_creator = CellImageSegmentor(return_channels=('green','red','blue'))
print (segment_creator)

for cell_id, data_path_collection in local_imgs.items():
    if not segment_creator.already_in_db_(cell_id):
        segment_creator.transform(data_path_collection).write_entry(cell_id)
#    segment_creator.transform(data_path_collection).write_entry(cell_id)

cellsegment_dataset = CellImageSegmentContrastDataset(cell_image_segmentor=segment_creator,
                                                      data_label_file='./data_tmp/train.csv')
dloader = DataLoader(cellsegment_dataset, batch_size=16, shuffle=True)
for xxx in dloader:
    print (xxx)
    raise RuntimeError