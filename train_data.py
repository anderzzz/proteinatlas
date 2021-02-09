'''Accessors of the image data

'''
from os import listdir
from enum import Enum

import numpy as np
from numpy import ndarray
from skimage.io import imread
import pandas as pd

class ImgMetaDataError(Exception):
    pass

class ImgNotNumpyArrayError(Exception):
    pass

class ImgNotGrayScaleError(Exception):
    pass

class ImgNotAllowedShape(Exception):
    pass


class ImgMetaData(Enum):
    suffix = ['blue', 'green', 'red', 'yellow']
    staining = ['nuclei', 'protein_of_interest', 'microtubule', 'ER']
    organelle = {0 : 'Nucleoplasm',
                 1 : 'Nuclear membrane',
                 2 : 'Nucleoli',
                 3 : 'Nucleoli fibrillar center',
                 4 : 'Nuclear speckles',
                 5 : 'Nuclear bodies',
                 6 : 'Endoplasmic reticulum',
                 7 : 'Golgi apparatus',
                 8 : 'Intermediate filaments',
                 9 : 'Actin filaments',
                 10 : 'Microtubules',
                 11 : 'Mitotic spindle',
                 12 : 'Centrosome',
                 13 : 'Plasma membrane',
                 14 : 'Mitochondria',
                 15 : 'Aggresome',
                 16 : 'Cytosol',
                 17 : 'Vesicles and punctate cytosolic patterns',
                 18 : 'Negative'}
    n_categories = 19
    allowed_img_sizes = [(1728, 1728), (2048, 2048), (3072, 3072)]

    @classmethod
    def semantic_from_label(cls, label):
        try:
            return cls.staining.value[cls.suffix.value.index(label)]
        except ValueError:
            raise ImgMetaDataError('Unknown label: {}'.format(label))

    @classmethod
    def label_from_semantic(cls, semantic):
        try:
            return cls.suffix.value[cls.staining.value.index(semantic)]
        except ValueError:
            raise ImgMetaDataError('Unknown semantic: {}'.format(semantic))


class ImgDataAccessorFactory(object):
    '''Bla bla

    '''
    def __init__(self):
        self._creators = {}

    def register_src_type(self, src_type, creator):
        self._creators[src_type] = creator

    def create(self, src_type, **kwargs):
        accessor = self._creators.get(src_type)
        if not accessor:
            raise ValueError('No accessor registered for data source type {}'.format(src_type))

        return accessor(**kwargs)


class ImgDataFromLocalDisk(object):
    '''Bla bla

    '''
    def __init__(self, src_dir, img_suffix='png'):
        self.src_dir = src_dir
        self.img_suffix = img_suffix
        len_img_suffix = len(img_suffix) + 1
        self.src_dir_img_content = [fname[:-len_img_suffix] for fname in listdir(self.src_dir) \
                                                               if '.{}'.format(img_suffix) in fname[-len_img_suffix:]]

        cell_ids_by_suffix = {}
        for suffix in ImgMetaData.suffix.value:
            formatted_suffix = '_{}'.format(suffix)
            cell_ids = [cell_id.replace(formatted_suffix, '') for cell_id in self.src_dir_img_content if formatted_suffix in cell_id]
            cell_ids_by_suffix[suffix] = cell_ids

        for suff1, v1 in cell_ids_by_suffix.items():
            for suff2, v2 in cell_ids_by_suffix.items():
                diff_12 = set(v1) - set(v2)
                diff_21 = set(v2) - set(v1)

                if len(diff_12) > 0:
                    raise ImgMetaDataError('Cell images of subtype {} without subtype {} found: {}'.format(suff1, suff2, diff_12))
                if len(diff_21) > 0:
                    raise ImgMetaDataError('Cell images of subtype {} without subtype {} found: {}'.format(suff2, suff1, diff_21))

        self.cell_ids = cell_ids_by_suffix[ImgMetaData.suffix.value[0]]

    def __getitem__(self, cell_id):
        if not cell_id in self.cell_ids:
            raise KeyError('Unknown cell id: {}'.format(cell_id))

        full_paths = {}
        for suffix in ImgMetaData.suffix.value:
            full_path = '{}/{}_{}.{}'.format(self.src_dir, cell_id, suffix, self.img_suffix)
            full_paths[suffix] = full_path

            staining = ImgMetaData.semantic_from_label(suffix)
            full_paths[staining] = full_path

        return full_paths

    def keys(self):
        return self.cell_ids

    def items(self):
        for cell_id in self.cell_ids:
            yield cell_id, self[cell_id]


factory = ImgDataAccessorFactory()
factory.register_src_type('local disk', ImgDataFromLocalDisk)


class ImgDataRetriever(object):
    '''Bla bla

    '''
    def __init__(self, img_reader_function=None, img_reader_function_kwargs={},
                 img_postprocessor=None, img_postprocessor_kwargs={}, img_postprocessor_returns_image=False):
        self.img_reader_function = img_reader_function
        self.img_reader_function_kwargs = img_reader_function_kwargs
        self.img_postprocessor = img_postprocessor
        self.img_postprocessor_kwargs = img_postprocessor_kwargs
        self.img_postprocessor_returns_image = img_postprocessor_returns_image

    def retrieve(self, img_ref):
        '''Bla bla

        '''
        _img = self.img_reader_function(img_ref, **self.img_reader_function_kwargs)
        if not self.img_postprocessor is None:
            ret_value = self.img_postprocessor(_img, **self.img_postprocessor_kwargs)
            if self.img_postprocessor_returns_image:
                _img = ret_value

        print (_img.max())
        return _img


def _check_type_dimension(arr):
    '''Bla bla

    '''
    if not isinstance(arr, ndarray):
        raise ImgNotNumpyArrayError('Image retrieved not an instance of the numpy array format')

    if not len(arr.shape) == 2:
        raise ImgNotGrayScaleError('Image retrieved not gray scale')

    if not arr.shape in ImgMetaData.allowed_img_sizes.value:
        raise ImgNotAllowedShape('Image retrieved not of allowed shape: {}'.format(arr.shape))

def _rescale_max(arr, max_val=255):
    '''Bla bla

    '''
    arr_tmp = arr.astype('float64')
    arr_tmp = np.round(arr_tmp * max_val / arr.max())
    return arr_tmp.astype('uint8')

def _rescale_check(arr):
    '''Bla bla

    '''
    _check_type_dimension(arr)
    return _rescale_max(arr)

skimage_img_retriever = ImgDataRetriever(img_reader_function=imread,
                                         img_postprocessor=_check_type_dimension)
skimage_img_retriever_rescaler = ImgDataRetriever(img_reader_function=imread,
                                                  img_postprocessor=_rescale_check,
                                                  img_postprocessor_returns_image=True)

def parse_labels(path, n_categories=ImgMetaData.n_categories.value):
    '''Parse CSV with weak cell class labels

    '''
    df = pd.read_csv(path)

    labels = df['Label'].str.split('|')
    data = labels.apply(lambda row: [1 if str(n) in row else 0 for n in range(n_categories)])
    data = pd.DataFrame(data.to_list(), columns=range(n_categories))
    df = df.join(data).drop(columns='Label').set_index('ID')

    return df