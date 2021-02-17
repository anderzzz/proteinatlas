'''Dataset and Dataloader for PyTorch after segmentation

'''
import pandas
import numpy as np

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from skimage.util import random_noise

from train_data import parse_labels, skimage_img_retriever_rescaler, ImgMetaData, image_factory
from segmentor import ConfocalNucleusSweepAreaMasker, ConfocalNucleusAreaMasker, \
                      ConfocalNucleusSweepSegmentor, \
                      ConfocalCellAreaMasker, ConfocalCellSegmentor
from shaper import ImageShapeMaker
from mask_coco_encoder import encode_binary_mask

MEAN = [110]
STD = [50]

class CellImageSegmentsTransform:
    '''Bla bla

    '''
    def __init__(self,
                 return_channels=('protein_of_interest',),
                 min_size_nucleus_object=10000,
                 min_hole_allowed=5000,
                 edge_width=50,
                 min_size_nucleus_object_at_edge=1000,
                 min_cell_allowed=10000,
                 max_edge_area_frac=0.50,
                 min_cell_luminosity=10,
                 output_tensor=False
                 ):

        if all([data_channel_name in ImgMetaData.suffix.value + ImgMetaData.staining.value for data_channel_name in return_channels]):
            self.return_channels = return_channels
        else:
            raise ValueError('Given data return channels not subset of image staining and suffix constants')

        self.min_size_nucleus_object = min_size_nucleus_object
        self.min_hole_allowed = min_hole_allowed
        self.edge_width = edge_width
        self.min_size_nucleus_object_at_edge = min_size_nucleus_object_at_edge
        self.min_cell_allowed = min_cell_allowed
        self.max_edge_area_frac = max_edge_area_frac
        self.min_cell_luminosity = min_cell_luminosity
        self.output_tensor = output_tensor

        maskers_sweep_nuc = [
            ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever_rescaler,
                                      edge_width=None,
                                      body_luminosity=50,
                                      body_object_area=self.min_size_nucleus_object,
                                      body_hole_area=self.min_hole_allowed),
            ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever_rescaler,
                                      edge_width=None,
                                      body_luminosity=30,
                                      body_object_area=self.min_size_nucleus_object,
                                      body_hole_area=self.min_hole_allowed),
            ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever_rescaler,
                                      edge_width=None,
                                      body_luminosity=20,
                                      body_object_area=self.min_size_nucleus_object,
                                      body_hole_area=self.min_hole_allowed),
            ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever_rescaler,
                                      edge_width=None,
                                      body_luminosity=10,
                                      body_object_area=self.min_size_nucleus_object,
                                      body_hole_area=self.min_hole_allowed),
            ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever_rescaler,
                                      edge_width=self.edge_width,
                                      body_luminosity=255,
                                      body_object_area=self.min_size_nucleus_object,
                                      body_hole_area=self.min_hole_allowed,
                                      edge_luminosity=5,
                                      edge_object_area=self.min_size_nucleus_object_at_edge,
                                      edge_hole_area=self.min_hole_allowed)
            ]
        self.masker_nuc = ConfocalNucleusSweepAreaMasker(img_retriever=skimage_img_retriever_rescaler,
                                                         maskers_sweep=maskers_sweep_nuc)
        self.segmentor_nuc = ConfocalNucleusSweepSegmentor(img_retriever=skimage_img_retriever_rescaler)
        self.masker_cell = ConfocalCellAreaMasker(img_retriever=skimage_img_retriever_rescaler,
                                     body_luminosity=self.min_cell_luminosity, body_object_area=100, body_hole_area=100)
        self.segmentor_cell = ConfocalCellSegmentor(img_retriever=skimage_img_retriever_rescaler)
        self.shaper_cell = ImageShapeMaker(img_retriever=skimage_img_retriever_rescaler)

    def __call__(self, data_path_collection):

        img_nuc = data_path_collection['nuclei']
        img_er = data_path_collection['ER']
        img_tube = data_path_collection['microtubule']

        #
        # Construct cell nuclei segments
        self.masker_nuc.make_mask_sweep_(img_nuc)
        self.segmentor_nuc.make_segments_(img_nuc, self.masker_nuc.maskers_sweep)
        self.masker_nuc.infer_mask_from_segments_(self.segmentor_nuc.segments)

        #
        # Construct initial generous cell segments
        self.masker_cell.make_mask_(img_tube, self.masker_nuc.mask)
        self.segmentor_cell.make_segments_(img_er, self.masker_cell.mask, self.masker_nuc.mask, self.segmentor_nuc.segments)

        #
        # Discard cell segments that by heuristics are not well described
        small_area_segments = [cell_counter for cell_counter, area in \
                               self.segmentor_cell.get_area_segments().items() \
                               if area < self.min_cell_allowed]
        segments_mostly_on_edge = [cell_counter for cell_counter, area_frac in \
                                   self.segmentor_cell.get_segments_areafraction_on_edge(self.edge_width).items() \
                                   if area_frac > self.max_edge_area_frac]
        for cell_counter in small_area_segments + segments_mostly_on_edge:
            self.segmentor_cell.del_segment(cell_counter)

        #
        # Modify cell segments such that they contain no holes
        self.segmentor_cell.fill_holes()

        self.segments_coco = {}
        for cell_counter, mask_segment in self.segmentor_cell.items():
            self.segments_coco[cell_counter] = encode_binary_mask(mask_segment)

        #
        # Reshape image to multiple images fitted to the cell segments and stack selected image channels
        imgs_cell = {}
        for channel in self.return_channels:
            self.shaper_cell.apply_to(data_path_collection[channel], self.segmentor_cell.mask_segment).cut_square()
            imgs_channel = self.shaper_cell.imgs_reshaped.copy()

            for cell_counter, img_channel in imgs_channel.items():
                channels_container = imgs_cell.setdefault(cell_counter, [])
                channels_container.append(img_channel)
                imgs_cell[cell_counter] = channels_container

        return {cell_counter: np.stack(channels_container) for cell_counter, channels_container in imgs_cell.items()}

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class CellImageSegmentOneClassDataset(IterableDataset):
    '''Bla bla

    '''
    def __init__(self,
                 data_source_type='local disk',
                 data_source_folder=None,
                 data_label_file=None,
                 cell_id_subset=None,
                 batch_size=64,
                 square_size=224,
                 gray_noise_range=0.05):
        super(CellImageSegmentOneClassDataset, self).__init__()

        self.data_source_type = data_source_type
        self.data_source_folder = data_source_folder
        self.data_label_file = data_label_file
        self.cell_id_subset = cell_id_subset
        self.batch_size = batch_size
        self.gray_noise_range = gray_noise_range
        self.square_size = square_size

        self.local_imgs = image_factory.create(self.data_source_type, folder=self.data_source_folder)
        if not self.data_label_file is None:
            self.df_label = parse_labels(self.data_label_file)
        else:
            raise ValueError('Ground truth labels file missing')
        self.cell_id_ok = lambda x: True if self.cell_id_subset is None else (True if x in self.cell_id_subset else False)

        self.img_batch = CellImageSegmentBatch(batch_size)

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

        self.segmentor_shaper_transform = CellImageSegmentsTransform()

    def _speckle_noise(self, img):
        return random_noise(img, mode='speckle', mean=0, var=self.gray_noise_range, clip=True)

    def _add_to_batch(self, data_path_collection):
        imgs_sq_segment = self.segmentor_shaper_transform(data_path_collection).values()
        imgs_sq_segment = [torch.tensor(x, dtype=torch.float32) for x in imgs_sq_segment]
        imgs_sq_segment = [self.resizer(x) for x in imgs_sq_segment]
        imgs_sq_batch = torch.stack(imgs_sq_segment)
        imgs_sq_batch = self.train_aug_transform(imgs_sq_batch)
        self.img_batch.extend(imgs_sq_batch)

    def __iter__(self):
        for cell_id, data_path_collection in self.local_imgs.items():
            if self.cell_id_ok(cell_id):
#                self._add_to_batch(data_path_collection)
                imgs_sq_segment = self.segmentor_shaper_transform(data_path_collection).values()
                imgs_sq_segment = [torch.tensor(x, dtype=torch.float32) for x in imgs_sq_segment]
                imgs_sq_segment = [self.resizer(x) for x in imgs_sq_segment]
                imgs_sq_batch = torch.stack(imgs_sq_segment)
                imgs_sq_batch_pos, imgs_sq_batch_contrast = self.train_aug_transform(imgs_sq_batch)
                self.img_batch.extend(imgs_sq_batch)

                label = self.df_label.loc[cell_id]
                print (label)


            if self.img_batch.is_batch_full():
                yield self.img_batch.pop()

        else:
            yield self.img_batch.pop()

class CellImageSegmentBatch:
    '''Bla bla

    '''
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.container = None

    def is_batch_full(self):
        return self.container.shape[0] >= self.batch_size

    def extend(self, payload):
        if self.container is None:
            self.container = payload
        else:
            self.container = torch.cat([self.container, payload], dim=0)

    def pop(self):
        content_return, content_remainder = torch.split(self.container, self.batch_size)
        self.container = content_remainder
        return content_return