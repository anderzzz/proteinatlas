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
                 gray_noise_range=0.05):
        super(CellImageSegmentContrastDataset, self).__init__()

        self.cell_image_segmentor = cell_image_segmentor
        self.data_label_file = data_label_file
        self.gray_noise_range = gray_noise_range
        self.square_size = square_size

        self.cell_image_segmentor.build_toc()

        if not self.data_label_file is None:
            self.df_label = parse_labels(self.data_label_file)
        else:
            raise ValueError('Ground truth labels file missing')

        random_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.square_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomApply([
            #    transforms.Lambda(self._speckle_noise)
            #], p=0.8),
            #transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        self.train_aug_transform = TwoCropTransform(random_transform)
        self.resizer = transforms.Resize(size=self.square_size)

    def _speckle_noise(self, img):
        return random_noise(img, mode='speckle', mean=0, var=self.gray_noise_range, clip=True)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()

        data = self.cell_image_segmentor.inspect_entry(item)
        label = torch.tensor(self.df_label.loc[data['cell_id']].tolist())

        images = self.cell_image_segmentor[item]
        images = self.resizer(images)
        transformed_images = self.train_aug_transform(images)
        images = torch.stack(transformed_images)

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
                 gray_noise_range=0.05):
        super().__init__(cell_image_segmentor=cell_image_segmentor, data_label_file=data_label_file,
                         square_size=square_size, gray_noise_range=gray_noise_range)

        self.positive_one_class = positive_one_class

    def __getitem__(self, item):

        label, images = super().__getitem__(item)
        return label[self.positive_one_class], images