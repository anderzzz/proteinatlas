'''Bla bla

'''
import os
import json
import numpy as np
import torch
from mask_coco_encoder import encode_binary_mask

from train_data import ImgMetaData, skimage_img_retriever_rescaler

from _segmentor import ConfocalNucleusSweepSegmentor, ConfocalNucleusAreaMasker, ConfocalNucleusSweepAreaMasker, \
    ConfocalCellAreaMasker, ConfocalCellSegmentor
from _shaper import ImageShapeMaker

class CellImageSegmentor(object):
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
                 save_folder='_tmp_save_segments',
                 save_folder_db='saved_transformed_segments.json'
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
        self.save_folder = save_folder
        self.save_folder_db = save_folder_db

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

        if not os.path.isdir(self.save_folder):
            try:
                os.mkdir('{}/{}'.format(os.getcwd(), self.save_folder))
            except OSError:
                raise OSError('Failed to create temporary directory {}/{}'.format(os.getcwd(), self.save_folder))

            self.reset()

    def transform(self, data_path_collection):

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

        self.img_segments = {cell_counter: np.stack(channels_container) for cell_counter, channels_container in imgs_cell.items()}

        return self

    def write_entry(self, cell_id, suffix='pt'):
        '''Bla bla

        '''
        if self.save_folder is None:
            raise RuntimeError('Save folder not set during initialization')

        entries = {}
        for cell_counter, cell_img_sq in self.img_segments.items():
            file_path = '{}/{}.{}.{}'.format(self.save_folder, cell_id, cell_counter, suffix)
            tensor = torch.tensor(cell_img_sq, dtype=torch.float32)
            torch.save(tensor, file_path)

            new_entry = {
                'file_path' : '{}'.format(file_path),
                'n_channels' : '{}'.format(tensor.shape[0]),
                'height' : '{}'.format(tensor.shape[1]),
                'width' : '{}'.format(tensor.shape[2]),
                'coco_mask' : '{}'.format(self.segments_coco[cell_counter])
            }
            for k_channel, channel_name in enumerate(self.return_channels):
                new_entry['channel_{}'.format(k_channel)] = channel_name

            entries['{}'.format(cell_counter)] = new_entry

        data = self._read_db()
        data['segments'][cell_id] = entries
        with open('{}/{}'.format(self.save_folder, self.save_folder_db), 'w') as json_db:
            json.dump(data, json_db)

    def items(self):
        '''Bla bla

        '''
        data = self._read_db()
        for cell_id, cell_data in data['segments'].items():
            for cell_counter, cell_segment_data in cell_data.items():
                tensor = torch.load(cell_segment_data['file_path'])
                yield (cell_id, cell_counter), tensor

    def keys(self):
        '''Bla bla

        '''
        data = self._read_db()
        return [(cell_id, cell_counter) for cell_id, subdata in data['segments'].items()
                                            for cell_counter in subdata.keys()]

    def __getitem__(self, item):
        '''Bla bla

        '''
        data = self._read_db()
        try:
            fp = data['segments'][item[0]][item[1]]['file_path']
        except KeyError:
            raise KeyError('No entry for {} found in transformed database. Have you run `transform` method on this cell image?')
        return torch.load(fp)

    def inspect_entry(self, item):
        '''Bla bla

        '''
        data = self._read_db()
        try:
            entry = data['segments'][item[0]][item[1]]
        except KeyError:
            raise KeyError('No entry for {} found in transformed database. Have you run `transform` method on this cell image?')
        return entry

    def _read_db(self):
        '''Bla bla

        '''
        with open('{}/{}'.format(self.save_folder, self.save_folder_db)) as f_json:
            data = json.load(f_json)
        return data

    def reset(self):
        '''Bla bla

        '''
        for tmp_file in os.listdir(self.save_folder):
            os.remove(tmp_file)
        with open('{}/{}'.format(self.save_folder, self.save_folder_db), 'w') as f_json:
            json.dump({'segments' : {}}, f_json)
