'''Test runs

'''
from train_data import parse_labels, factory, skimage_img_retriever
from segmentor import ConfocalNucleusAreaMasker, ConfocalNucleusSegmentor, \
                      ConfocalCellAreaMasker, ConfocalCellSegmentor, \
                      ConfocalNucleusSweepSegmentor, ConfocalNucleusSweepAreaMasker
from shaper import ImageShapeMaker
from visualiser import Visualiser

local_imgs = factory.create('local disk', src_dir='./data_tmp')


maskers_sweep_nuc = [
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=50, body_object_area=10000, body_hole_area=5000),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=30, body_object_area=10000, body_hole_area=5000),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=10, body_object_area=10000, body_hole_area=5000),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=5, body_object_area=10000, body_hole_area=5000),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=10,
                              body_luminosity=5, body_object_area=10000, body_hole_area=5000,
                              edge_luminosity=20, edge_object_area=1000, edge_hole_area=5000),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=20,
                              body_luminosity=5, body_object_area=10000, body_hole_area=5000,
                              edge_luminosity=20, edge_object_area=1000, edge_hole_area=5000),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=50,
                              body_luminosity=5, body_object_area=10000, body_hole_area=5000,
                              edge_luminosity=20, edge_object_area=1000, edge_hole_area=5000)
               ]
masker_nuc = ConfocalNucleusSweepAreaMasker(img_retriever=skimage_img_retriever,
                                            maskers_sweep=maskers_sweep_nuc)
segmentor_nuc = ConfocalNucleusSweepSegmentor(img_retriever=skimage_img_retriever)

masker_cell = ConfocalCellAreaMasker(img_retriever=skimage_img_retriever,
                                     body_luminosity=7, body_object_area=100, body_hole_area=100)
segmentor_cell = ConfocalCellSegmentor(img_retriever=skimage_img_retriever)

shaper_cell = ImageShapeMaker(img_retriever=skimage_img_retriever)
viz = Visualiser(cmap='gray', cmap_set_under='green')

df_labels = parse_labels('./data_tmp/train.csv')
for cell_id, data_path_collection in local_imgs.items():

    if not '0020af' in cell_id:
        continue

    img_nuc = data_path_collection['nuclei']
    img_er = data_path_collection['ER']
    img_tube = data_path_collection['microtubule']
    img_prot = data_path_collection['green']

    masker_nuc.make_mask_sweep_(img_nuc)
    segmentor_nuc.make_segments_(img_nuc, masker_nuc.maskers_sweep)
    masker_nuc.infer_mask_from_segments_(segmentor_nuc.segments)
    viz.show_segments_overlay(skimage_img_retriever.retrieve(img_nuc), segmentor_nuc.segments)

    masker_cell.make_mask_(img_tube, masker_nuc.mask)
    segmentor_cell.make_segments_(img_er, masker_cell.mask, masker_nuc.mask, segmentor_nuc.segments)

    segmentor_cell.del_segments(segmentor_nuc.get_segments_on_edge())
    segmentor_cell.fill_holes()

    viz.show_segments_overlay(skimage_img_retriever.retrieve(img_tube), segmentor_cell.segments)

    shaper_cell.apply_to(img_prot, segmentor_cell.mask_segment).outline()
    percell_prot = shaper_cell.imgs_reshaped.copy()
    shaper_cell.apply_to(img_er, segmentor_cell.mask_segment).outline()
    percell_er = shaper_cell.imgs_reshaped.copy()

    for cell_counter in percell_prot.keys():
        viz.show_(percell_prot[cell_counter], percell_er[cell_counter])
    #viz.show_(*tuple(percell_prot.values()))

    print (df_labels.loc[cell_id])


    raise RuntimeError
