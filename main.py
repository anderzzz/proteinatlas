'''Test runs

'''
from skimage.io import imread

from train_data import parse_labels, factory
from segmentor import ConfocalCellSegmentor, ConfocalNucleusSegmentor

local_imgs = factory.create('local disk', src_dir='./data_tmp')
segmentor_nuc = ConfocalNucleusSegmentor(img_data_accessor=local_imgs, reader_func=imread)
segmentor_cell = ConfocalCellSegmentor(img_data_accessor=local_imgs, reader_func=imread, nucleus_segmentor=segmentor_nuc)

df_labels = parse_labels('./data_tmp/train.csv')

for cell_id, data_path_collection in local_imgs.items():

    img_nuc = data_path_collection['nuclei']
    img_er = data_path_collection['ER']

    segmentor_cell.segment(img_er, img_nuc)




    raise RuntimeError
