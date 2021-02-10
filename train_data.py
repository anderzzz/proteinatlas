'''Objects for accessing the image raw data of cells from the human protein atlas.

The first set of objects manage the image file paths and their meta data. The objects are best accessed using
a factory object. Hence,

> from train_data import image_factory
> img_files = image_factory.create('kaggle notebook', folder='hpa-single-cell-image-classification/train')

creates a dictionary-like object `img_files` that associates a cell ID key with a dictionary containing the
file paths to the images with different staining.

The second set of objects manage the image retrieval given path. Already created retriever objects are:

`skimage_img_retriever_rescaler` : reads gray scale image file, checks for basic errors, and rescale intensity such
that maximum value is 255, and returns Numpy array representation

`skimage_img_retriever` : reads gray scale image file, checks for basic errors, and returns Numpy array representation.

The third set of objects manage the training data ground truth labels. That includes the convenience function
`parse_labels`

'''
from os import listdir
from enum import Enum

import numpy as np
from numpy import ndarray
from skimage.io import imread
import pandas as pd

#
# Section 1: Factory method to retrieve raw data file paths
#
class ImgMetaDataError(Exception):
    pass


class ImgMetaData(Enum):
    '''Enumerations of image content meta data.

    Attributes:
        suffix : The colours of the cell staining
        staining : The object strained, in identical order to `suffix`
        organelle : Integer class label and corresponding cell organelle
        n_categories : Number of unique organelle class labels
        allowed_img_sizes : Expected dimensions of images

    '''
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
    '''Factory method to provide common interface to raw image data access from diverse sources

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


class _ImgDataFromSomewhere(object):
    '''Parent class for raw image data accessor

    '''
    def __init__(self, folder, img_suffix='png'):
        self.folder = folder
        self.img_suffix = img_suffix

        self.cell_ids = None

    def make_cell_img_ids(self, imgs):
        '''Collect all cell image IDs

        Raises:
            ImgMetaDataError : In case cell id found for which not all image types (blue, green, red, yellow) are found

        '''
        cell_ids_by_suffix = {}
        for suffix in ImgMetaData.suffix.value:
            formatted_suffix = '_{}'.format(suffix)
            cell_ids = [cell_id.replace(formatted_suffix, '') for cell_id in imgs if formatted_suffix in cell_id]
            cell_ids_by_suffix[suffix] = cell_ids

        for suff1, v1 in cell_ids_by_suffix.items():
            for suff2, v2 in cell_ids_by_suffix.items():
                diff_12 = set(v1) - set(v2)
                diff_21 = set(v2) - set(v1)

                if len(diff_12) > 0:
                    raise ImgMetaDataError('Cell images of subtype {} without subtype {} found: {}'.format(suff1, suff2, diff_12))
                if len(diff_21) > 0:
                    raise ImgMetaDataError('Cell images of subtype {} without subtype {} found: {}'.format(suff2, suff1, diff_21))

        return cell_ids_by_suffix[ImgMetaData.suffix.value[0]]

    def image_types_collector(self, cell_id, full_path_formatter):
        '''Collect all paths to images associated with given cell ID

        Args:
            cell_id : Cell id to collect file paths for
            full_path_formatter : Function that given the cell ID and the colour suffix returns a formatted string
                appropriate to how the data bucket is organized

        '''
        if not cell_id in self.cell_ids:
            raise KeyError('Unknown cell id: {}'.format(cell_id))

        full_paths = {}
        for suffix in ImgMetaData.suffix.value:
            full_path = full_path_formatter(cell_id, suffix)
            full_paths[suffix] = full_path

            staining = ImgMetaData.semantic_from_label(suffix)
            full_paths[staining] = full_path

        return full_paths

    def __getitem__(self, cell_id):
        raise NotImplementedError('Should be overridden in child class')

    def keys(self):
        return self.cell_ids

    def items(self):
        for cell_id in self.cell_ids:
            yield cell_id, self[cell_id]


class ImgDataFromLocalDisk(_ImgDataFromSomewhere):
    '''Factory class for local disk storage of image raw data

    Args:
        folder : parent folder string, e.g. "./my_data"
        img_suffix : image file suffix (default: "png")

    '''
    def __init__(self, folder, img_suffix='png'):
        super().__init__(folder, img_suffix)

        len_img_suffix = len(self.img_suffix) + 1
        imgs = [fname[:-len_img_suffix] for fname in listdir(self.folder) \
                                            if '.{}'.format(img_suffix) in fname[-len_img_suffix:]]

        self.cell_ids = self.make_cell_img_ids(imgs)

    def file_path_formatter(self, cell_id, suffix):
        '''Formatter of path, given cell_id and file colour suffix

        '''
        return '{}/{}_{}.{}'.format(self.folder, cell_id, suffix, self.img_suffix)

    def __getitem__(self, cell_id):
        return self.image_types_collector(cell_id, self.file_path_formatter)


class ImgDataFromKaggleNotebook(_ImgDataFromSomewhere):
    '''Factory class for Kaggle Notebook storage of image raw data

    Args:
        folder : parent folder string, e.g. "some_competition_data/train"
        img_suffix : image file suffix (default: "png")

    '''
    kaggle_input_root = '/kaggle/input'

    def __init__(self, folder, img_suffix='png'):
        super().__init__(folder, img_suffix)

        len_img_suffix = len(self.img_suffix) + 1
        imgs = [fname[:-len_img_suffix] for fname in listdir('{}/{}'.format(self.kaggle_input_root, self.folder)) \
                                            if '.{}'.format(img_suffix) in fname[-len_img_suffix:]]

        self.cell_ids = self.make_cell_img_ids(imgs)

    def file_path_formatter(self, cell_id, suffix):
        '''Formatter of path, given cell_id and file colour suffix

        '''
        return '{}/{}/{}_{}.{}'.format(self.kaggle_input_root, self.folder, cell_id, suffix, self.img_suffix)

    def __getitem__(self, cell_id):
        return self.image_types_collector(cell_id, self.file_path_formatter)


image_factory = ImgDataAccessorFactory()
image_factory.register_src_type('local disk', ImgDataFromLocalDisk)
image_factory.register_src_type('kaggle notebook', ImgDataFromKaggleNotebook)

#
# Section 2: Extracting the image data at a given path, along with basic error checking and simple post processing
#
class ImgNotNumpyArrayError(Exception):
    pass

class ImgNotGrayScaleError(Exception):
    pass

class ImgNotAllowedShape(Exception):
    pass


class ImgDataRetriever(object):
    '''Retrieve image data at a path, plus error checking and basic post processing

    Args:
        img_reader_function : function that can read the image file, like `imread` from `skimage` library
        img_reader_function_kwargs : additional arguments to the image reader function

    '''
    def __init__(self, img_reader_function=None, img_reader_function_kwargs={},
                 img_postprocessor=None, img_postprocessor_kwargs={}, img_postprocessor_returns_image=False):
        self.img_reader_function = img_reader_function
        self.img_reader_function_kwargs = img_reader_function_kwargs
        self.img_postprocessor = img_postprocessor
        self.img_postprocessor_kwargs = img_postprocessor_kwargs
        self.img_postprocessor_returns_image = img_postprocessor_returns_image

    def retrieve(self, img_ref):
        '''Retrieve, and optionally error check and post-process image raw data

        '''
        _img = self.img_reader_function(img_ref, **self.img_reader_function_kwargs)
        if not self.img_postprocessor is None:
            ret_value = self.img_postprocessor(_img, **self.img_postprocessor_kwargs)
            if self.img_postprocessor_returns_image:
                _img = ret_value

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

#
# Section 3: Objects dealing with the tabular training data labels
#
def parse_labels(path, n_categories=ImgMetaData.n_categories.value):
    '''Parse CSV with weak cell class labels

    '''
    df = pd.read_csv(path)

    labels = df['Label'].str.split('|')
    data = labels.apply(lambda row: [1 if str(n) in row else 0 for n in range(n_categories)])
    data = pd.DataFrame(data.to_list(), columns=range(n_categories))
    df = df.join(data).drop(columns='Label').set_index('ID')

    return df