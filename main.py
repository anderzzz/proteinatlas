'''Test runs

'''
from skimage.io import imread

from train_data import parse_labels, factory
from segmentor import ConfocalCellSegmentor, ConfocalNucleusSegmentor
from shaper import ImageShapeMaker
from visualiser import Visualiser

local_imgs = factory.create('local disk', src_dir='./data_tmp')
segmentor_nuc = ConfocalNucleusSegmentor(img_data_accessor=local_imgs, reader_func=imread)
segmentor_cell = ConfocalCellSegmentor(img_data_accessor=local_imgs, reader_func=imread,
                                       nucleus_segmentor=segmentor_nuc)
shaper_cell = ImageShapeMaker(img_data_accessor=local_imgs, reader_func=imread)
viz = Visualiser(cmap='gray')

df_labels = parse_labels('./data_tmp/train.csv')

for cell_id, data_path_collection in local_imgs.items():

    img_nuc = data_path_collection['nuclei']
    img_er = data_path_collection['ER']
    img_prot = data_path_collection['green']

    segmentor_cell.segment(img_er, img_nuc).prune()

    shaper_cell.apply_to(img_prot, segmentor_cell.mask).cut_square()
    percell_prot = shaper_cell.cell_images_reshaped.copy()
    shaper_cell.apply_to(img_er, segmentor_cell.mask).cut_square()
    percell_er = shaper_cell.cell_images_reshaped.copy()

    #for cell_counter in percell_prot.keys():
    #    viz.plot_(percell_prot[cell_counter], percell_er[cell_counter])
    #viz.show_(*tuple(percell_er.values()))
    
    print (df_labels.loc[cell_id])


    raise RuntimeError
