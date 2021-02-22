'''Dataset and Dataloader for PyTorch after segmentation

'''
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage.util import random_noise

from train_data import parse_labels

MEAN = [110]
STD = [50]

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class CellImageSegmentContrastDataset(Dataset):
    '''Bla bla

    '''
    def __init__(self,
                 cell_image_segmentor=None,
                 data_label_file=None,
                 square_size=224,
                 crop_scale=(0.7,1.0),
                 gray_noise_range=0.20,
                 image_dtype=torch.float32):
        super(CellImageSegmentContrastDataset, self).__init__()

        self.cell_image_segmentor = cell_image_segmentor
        self.data_label_file = data_label_file
        self.gray_noise_range = gray_noise_range
        self.square_size = square_size
        self.crop_scale = crop_scale
        self.image_dtype = image_dtype

        self.cell_image_segmentor.build_toc()

        if not self.data_label_file is None:
            self.df_label = parse_labels(self.data_label_file)
        else:
            raise ValueError('Ground truth labels file missing')

        # The input tensor is in range 0 to 255. The transformations are:
        # 1. Scale tensor to range 0 to 1.
        # 2. Crop image to random size and aspect ration, followed by scaling back to input size
        # 3. Do horizontal flip with 0.5 probability
        # 4. Scale the magnitude of pixel value, scale from gaussian (clipped to range 0.0 to 1.0)
        random_transform = transforms.Compose([
            transforms.Normalize(mean=0.0, std=255.0),
            transforms.RandomResizedCrop(size=self.square_size, scale=self.crop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.Lambda(self._speckle_noise)
            ], p=0.8)
        ])
        self.train_aug_transform = TwoCropTransform(random_transform)
        self.resizer = transforms.Resize(size=self.square_size)

    def _speckle_noise(self, img):
        noise_added = random_noise(img.numpy(), mode='speckle', mean=0, var=self.gray_noise_range, clip=True)
        return torch.tensor(noise_added)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        data = self.cell_image_segmentor.inspect_entry(item)
        label = torch.tensor(self.df_label.loc[data['cell_id']].tolist())

        images = self.cell_image_segmentor[item]
        images = self.resizer(images)
        transformed_images = self.train_aug_transform(images)
        images = torch.stack(transformed_images)
        images = images.type(self.image_dtype)

        return label, images

    def __len__(self):
        return len(self.cell_image_segmentor)


class CellImageSegmentOneClassContrastDataset(CellImageSegmentContrastDataset):
    '''Bla bla

    '''
    def __init__(self,
                 positive_one_class,
                 cell_image_segmentor=None,
                 data_label_file=None,
                 square_size=224,
                 gray_noise_range=0.05,
                 image_dtype=torch.float32):
        super().__init__(cell_image_segmentor=cell_image_segmentor, data_label_file=data_label_file,
                         square_size=square_size, gray_noise_range=gray_noise_range,
                         image_dtype=image_dtype)

        self.positive_one_class = positive_one_class

    def __getitem__(self, item):

        label, images = super().__getitem__(item)
        return label[self.positive_one_class], images

    @property
    def positive_items(self):
        pos_cell_ids = self.df_label.loc[self.df_label[self.positive_one_class] == 1].index
        pos_mask = self.cell_image_segmentor.pd_toc['cell_id'].isin(pos_cell_ids)
        return self.cell_image_segmentor.pd_toc[pos_mask].index.to_list()

    @property
    def negative_items(self):
        neg_cell_ids = self.df_label.loc[self.df_label[self.positive_one_class] == 0].index
        neg_mask = self.cell_image_segmentor.pd_toc['cell_id'].isin(neg_cell_ids)
        return self.cell_image_segmentor.pd_toc[neg_mask].index.to_list()