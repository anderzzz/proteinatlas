'''Dataset and Dataloader for PyTorch after segmentation

'''
import pandas
import numpy as np

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from skimage.util import random_noise

from train_data import parse_labels, skimage_img_retriever_rescaler, ImgMetaData
from segmentor import ConfocalNucleusSweepAreaMasker, ConfocalNucleusAreaMasker, \
                      ConfocalNucleusSweepSegmentor, \
                      ConfocalCellAreaMasker, ConfocalCellSegmentor
from shaper import ImageShapeMaker
from mask_coco_encoder import encode_binary_mask

MEAN = 0.4
STD = 0.5

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
        img_prot = data_path_collection['green']

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

        self.segments_coco = []
        for cell_counter, mask_segment in self.segmentor_cell.items():
            self.segments_coco.append(encode_binary_mask(mask_segment))

        #
        # Reshape image to multiple images fitted to the cell segments
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

class CellSegmentGrayOneClassDataset(IterableDataset):
    '''Bla bla

    '''
    def __init__(self,
                 data_source_type='local disk',
                 data_source_folder=None,
                 data_label_folder=None,
                 batch_size=64,
                 square_size=224,
                 gray_noise_range=0.05):
        super(CellSegmentGrayDataset, self).__init__()

        self.data_source_type = data_source_type
        self.data_source_folder = data_source_folder
        self.data_label_folder = data_label_folder
        self.batch_size = batch_size
        self.gray_noise_range = gray_noise_range
        self.square_size = square_size

        if not self.data_label_folder is None:
            df_label = parse_labels(self.data_label_folder)

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.square_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.Lambda(self._speckle_noise)
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def _speckle_noise(self, img):
        random_noise(img, mode='speckle', mean=0, var=self.gray_noise_range, clip=True)

    def __iter__(self):
        raise NotImplementedError
        yield 'dude'
