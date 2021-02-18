'''Bla bla

'''
from torch.utils.data import DataLoader

from train_data import image_factory
from cellsegments import CellImageSegmentor
from dataset import CellImageSegmentContrastDataset

from supcontrast.resnet_big import SupConResNet
from supcontrast.losses import SupConLoss

local_imgs = image_factory.create('local disk', folder='./data_tmp')
segment_creator = CellImageSegmentor(return_channels=('green','red','blue'))
for cell_id, data_path_collection in local_imgs.items():
    if not segment_creator.already_in_db_(cell_id):
        segment_creator.transform(data_path_collection).write_entry(cell_id)
#    segment_creator.transform(data_path_collection).write_entry(cell_id)

cellsegment_dataset = CellImageSegmentContrastDataset(cell_image_segmentor=segment_creator,
                                                      data_label_file='./data_tmp/train.csv')
dloader = DataLoader(cellsegment_dataset, batch_size=16, shuffle=True)

n_epochs = 2
for epoch in range(n_epochs):

    for idx, (labels, images) in enumerate(dloader):

        print (images.shape)
        #start adding in main_supcon stuff
        raise RuntimeError