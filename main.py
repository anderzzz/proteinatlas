'''Test runs

'''
from skimage.io import imread

from train_data import parse_labels, factory
from segmentor import ConfocalNucleusAreaMasker, ConfocalNucleusSegmentor, \
                      ConfocalCellAreaMasker, ConfocalCellSegmentor
from shaper import ImageShapeMaker
from visualiser import Visualiser

local_imgs = factory.create('local disk', src_dir='./data_tmp')

masker_nuc = ConfocalNucleusAreaMasker(img_data_accessor=local_imgs, reader_func=imread,
                                       edge_width=10,
                                       body_luminosity=25, body_object_area=5000, body_hole_area=5000,
                                       edge_luminosity=25, edge_object_area=6, edge_hole_area=5000)
segmentor_nuc = ConfocalNucleusSegmentor(img_data_accessor=local_imgs, reader_func=imread)

masker_cell = ConfocalCellAreaMasker(img_data_accessor=local_imgs, reader_func=imread,
                                     body_luminosity=0, body_object_area=100, body_hole_area=100)
segmentor_cell = ConfocalCellSegmentor(img_data_accessor=local_imgs, reader_func=imread)

#segmentor_nuc = ConfocalNucleusSegmentor(img_data_accessor=local_imgs, reader_func=imread, edge_width=10)
#segmentor_cell = ConfocalCellSegmentor(img_data_accessor=local_imgs, reader_func=imread,
#                                       nucleus_segmentor=segmentor_nuc,
#                                       body_luminosity=30, body_hole_area=20000, body_object_area=10000)
shaper_cell = ImageShapeMaker(img_data_accessor=local_imgs, reader_func=imread)
viz = Visualiser(cmap='gray')

df_labels = parse_labels('./data_tmp/train.csv')

for cell_id, data_path_collection in local_imgs.items():

    if not '000a' in cell_id:
        continue

    img_nuc = data_path_collection['nuclei']
    img_er = data_path_collection['ER']
    img_tube = data_path_collection['red']
    img_prot = data_path_collection['green']

    masker_nuc.make_mask_(img_nuc)
    segmentor_nuc.make_segments_(img_nuc, masker_nuc.mask)

    masker_cell.make_mask_(img_tube, masker_nuc.mask)
    segmentor_cell.make_segments_(img_er, masker_cell.mask, masker_nuc.mask, segmentor_nuc.segments)

    raise RuntimeError

    shaper_cell.apply_to(img_prot, segmentor_cell.mask).outline()
    percell_prot = shaper_cell.cell_images_reshaped.copy()
    shaper_cell.apply_to(img_er, segmentor_cell.mask).outline()
    percell_er = shaper_cell.cell_images_reshaped.copy()

    for cell_counter in percell_prot.keys():
        viz.show_(percell_prot[cell_counter], percell_er[cell_counter])
    viz.show_(*tuple(percell_er.values()))

    print (df_labels.loc[cell_id])


    raise RuntimeError
