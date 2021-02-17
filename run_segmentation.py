'''Bla bla

'''
import pandas as pd
import torch
from torchvision.utils import save_image

from train_data import parse_labels, skimage_img_retriever_rescaler, ImgMetaData, image_factory
from dataset import CellImageSegmentsTransform

# Where to collect raw image data from
confocal_imgs = image_factory.create('local disk', folder='../data_tmp')

# The ground truth labels
df_labels = parse_labels('../data_tmp/train.csv')

# The segmentor that extracts individual cells from a confocal image
segmentor_shaper_transform = CellImageSegmentsTransform(
                                 return_channels=('green',)
)

coco_mask = {}
for cell_id, data_collection in confocal_imgs.items():
    cell_segments = segmentor_shaper_transform(data_collection)

    for cell_counter, cell_img_sq in cell_segments.items():
        save_image(torch.tensor(cell_img_sq, dtype=torch.float32), '../data_tmp_segments/{}_{}.png'.format(cell_id, cell_counter))
        coco_mask[(cell_id, cell_counter)] = segmentor_shaper_transform.segments_coco[cell_counter]

df_coco = pd.DataFrame(coco_mask.values(), index=pd.MultiIndex.from_tuples(coco_mask.keys(), names=['cell_id', 'cell_counter']), columns=['coco_mask'])
df_coco.to_csv('../data_tmp_segments/coco_mask.csv')
