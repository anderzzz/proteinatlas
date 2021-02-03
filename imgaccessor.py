'''Accessors of the image data

'''
from os import listdir
from enum import Enum
from collections import namedtuple

class ImgMetaDataError(Exception):
    pass


class ImgMetaData(Enum):
    suffix = ['blue', 'green', 'red', 'yellow']
    staining = ['nuclei', 'protein_of_interest', 'microtubule', 'ER']

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