'''Bla bla

'''
import os
import json
import numpy as np
import pandas as pd
import torch
from mask_coco_encoder import encode_binary_mask

from train_data import ImgMetaData, skimage_img_retriever_rescaler

from _segmentor import ConfocalNucleusSweepSegmentor, ConfocalNucleusAreaMasker, ConfocalNucleusSweepAreaMasker, \
    ConfocalCellAreaMasker, ConfocalCellSegmentor
from _shaper import ImageShapeMaker

def check_after_toc_build(f):
    def wrapper(*args):
        if args[0].pd_toc is None:
            raise RuntimeError('Before method {} is used, `build_toc` must be called')
        return f(*args)
    return wrapper

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

        self.db_master_id = 0
        self.primary_id_to_cell = {}
        self.pd_toc = None

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
                'id' : '{}'.format(self.db_master_id),
                'file_path' : '{}'.format(file_path),
                'n_channels' : '{}'.format(tensor.shape[0]),
                'height' : '{}'.format(tensor.shape[1]),
                'width' : '{}'.format(tensor.shape[2]),
                'coco_mask' : '{}'.format(self.segments_coco[cell_counter])
            }
            for k_channel, channel_name in enumerate(self.return_channels):
                new_entry['channel_{}'.format(k_channel)] = channel_name

            entries['{}'.format(cell_counter)] = new_entry

            self.primary_id_to_cell[self.db_master_id] = (cell_id, cell_counter)
            self.db_master_id += 1

        data = self._read_db()
        data['segments'][cell_id] = entries
        with open('{}/{}'.format(self.save_folder, self.save_folder_db), 'w') as json_db:
            json.dump(data, json_db)

    def build_toc(self):
        '''Bla bla

        '''
        data = self._read_db()
        df = pd.DataFrame.from_dict({(i,j,k) : data[i][j][k] for i in data.keys()
                                                             for j in data[i].keys()
                                                             for k in data[i][j].keys()}, orient='index')
        df = df.reset_index()
        df = df.rename(columns={'level_0' : 'entry_type', 'level_1' : 'cell_id', 'level_2' : 'cell_counter'}).set_index('id')
        df.index = df.index.astype(np.uint64)
        self.pd_toc = df

    @check_after_toc_build
    def items(self):
        '''Bla bla

        '''
        for dbid, row in self.pd_toc.iterrows():
            tensor = torch.load(row['file_path'])
            yield dbid, tensor

    @check_after_toc_build
    def keys(self):
        '''Bla bla

        '''
        return self.pd_toc.index.to_list()

    @check_after_toc_build
    def __len__(self):
        return len(self.keys())

    @check_after_toc_build
    def __getitem__(self, item):
        '''Bla bla

        '''
        return torch.load(self.pd_toc.loc[item]['file_path'])

    @check_after_toc_build
    def inspect_entry(self, item):
        '''Bla bla

        '''
        return dict(self.pd_toc.loc[item])

    def already_in_db_(self, cell_id):
        '''Bla bla

        '''
        data = self._read_db()
        return cell_id in data['segments'].keys()

    def reset(self):
        '''Bla bla

        '''
        self.db_master_id = 0
        self.primary_id_to_cell = {}
        for tmp_file in os.listdir(self.save_folder):
            os.remove(tmp_file)
        with open('{}/{}'.format(self.save_folder, self.save_folder_db), 'w') as f_json:
            json.dump({'segments' : {}}, f_json)

    def _read_db(self):
        '''Bla bla

        '''
        with open('{}/{}'.format(self.save_folder, self.save_folder_db)) as f_json:
            data = json.load(f_json)
        return data

