'''Bla bla

'''
import torch
from torch.utils.data import DataLoader

from train_data import image_factory
from cellsegments import CellImageSegmentor
from dataset import CellImageSegmentContrastDataset

from supcontrast.resnet_big import SupConResNet
from supcontrast.losses import SupConLoss

#
# Connect to raw data images
local_imgs = image_factory.create('local disk', folder='./data_tmp')

#
# Connect to image segmentor and execute to create smaller set of single cell images
segment_creator = CellImageSegmentor(return_channels=('green','red','blue'))
for cell_id, data_path_collection in local_imgs.items():
    if not segment_creator.already_in_db_(cell_id):
        segment_creator.transform(data_path_collection).write_entry(cell_id)
#    segment_creator.transform(data_path_collection).write_entry(cell_id)

#
# Connect to dataset over all available single cell images
cellsegment_dataset = CellImageSegmentContrastDataset(cell_image_segmentor=segment_creator,
                                                      data_label_file='./data_tmp/train.csv')
dloader = DataLoader(cellsegment_dataset, batch_size=16, shuffle=True)

#
# Define model and loss function
model = SupConResNet(name='resnet18', feat_dim=19)
criterion = SupConLoss()

#
# Training loop
model.train()
n_epochs = 2
for epoch in range(n_epochs):

    for idx, (labels, images) in enumerate(dloader):

        print (images.shape)
        view_1, view_2 = torch.unbind(images, dim=1)
        images = torch.cat([view_1, view_2], dim=0)

        features = model(images)
        f1, f2 = torch.split(features, [16, 16], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        criterion(features, labels)
        #start adding in main_supcon stuff
        raise RuntimeError