'''Test runs

'''
from train_data import parse_labels, image_factory, skimage_img_retriever
from segmentor import ConfocalNucleusAreaMasker, ConfocalNucleusSegmentor, \
                      ConfocalCellAreaMasker, ConfocalCellSegmentor, \
                      ConfocalNucleusSweepSegmentor, ConfocalNucleusSweepAreaMasker
from shaper import ImageShapeMaker
from visualiser import Visualiser
from mask_coco_encoder import encode_binary_mask

local_imgs = image_factory.create('local disk', folder='./data_tmp')

MIN_SIZE_NUCLEUS_OBJECT = 10000
MIN_HOLE_ALLOWED = 5000
EDGE_WIDTH = 50
MIN_SIZE_NUCLEUS_OBJECT_AT_EDGE = 1000
MIN_CELL_ALLOWED = 10000
MAX_EDGE_AREA_FRAC = 0.50
MIN_CELL_LUMINOSITY = 10

maskers_sweep_nuc = [
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=50,
                              body_object_area=MIN_SIZE_NUCLEUS_OBJECT, body_hole_area=MIN_HOLE_ALLOWED),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=30,
                              body_object_area=MIN_SIZE_NUCLEUS_OBJECT, body_hole_area=MIN_HOLE_ALLOWED),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=20,
                              body_object_area=MIN_SIZE_NUCLEUS_OBJECT, body_hole_area=MIN_HOLE_ALLOWED),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=10,
                              body_object_area=MIN_SIZE_NUCLEUS_OBJECT, body_hole_area=MIN_HOLE_ALLOWED),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=None,
                              body_luminosity=5,
                              body_object_area=MIN_SIZE_NUCLEUS_OBJECT, body_hole_area=MIN_HOLE_ALLOWED),
    ConfocalNucleusAreaMasker(img_retriever=skimage_img_retriever,
                              edge_width=EDGE_WIDTH,
                              body_luminosity=255,
                              body_object_area=MIN_SIZE_NUCLEUS_OBJECT, body_hole_area=MIN_HOLE_ALLOWED,
                              edge_luminosity=5,
                              edge_object_area=MIN_SIZE_NUCLEUS_OBJECT_AT_EDGE, edge_hole_area=MIN_HOLE_ALLOWED)
               ]
masker_nuc = ConfocalNucleusSweepAreaMasker(img_retriever=skimage_img_retriever,
                                            maskers_sweep=maskers_sweep_nuc)
segmentor_nuc = ConfocalNucleusSweepSegmentor(img_retriever=skimage_img_retriever)

masker_cell = ConfocalCellAreaMasker(img_retriever=skimage_img_retriever,
                                     body_luminosity=MIN_CELL_LUMINOSITY, body_object_area=100, body_hole_area=100)
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

    #
    # Construct cell nuclei segments
    #
    masker_nuc.make_mask_sweep_(img_nuc)
    segmentor_nuc.make_segments_(img_nuc, masker_nuc.maskers_sweep)
    masker_nuc.infer_mask_from_segments_(segmentor_nuc.segments)
    viz.show_segments_overlay(skimage_img_retriever.retrieve(img_nuc), segmentor_nuc.segments)

    #
    # Construct initial generous cell segments
    #
    masker_cell.make_mask_(img_tube, masker_nuc.mask)
    segmentor_cell.make_segments_(img_er, masker_cell.mask, masker_nuc.mask, segmentor_nuc.segments)

    #
    # Discard cell segments that by heuristics are not well described
    #
    small_area_segments = [cell_counter for cell_counter, area in \
                               segmentor_cell.get_area_segments().items() \
                               if area < MIN_CELL_ALLOWED]
    segments_mostly_on_edge = [cell_counter for cell_counter, area_frac in \
                                   segmentor_cell.get_segments_areafraction_on_edge(EDGE_WIDTH).items() \
                                   if area_frac > MAX_EDGE_AREA_FRAC]
    for cell_counter in small_area_segments + segments_mostly_on_edge:
        segmentor_cell.del_segment(cell_counter)

    #
    # Modify cell segments such that they contain no holes
    #
    segmentor_cell.fill_holes()
    for cell_counter, mask_segment in segmentor_cell.items():
        print (cell_counter)
        print (mask_segment.shape)
        print (encode_binary_mask(mask_segment))

    viz.show_segments_overlay(skimage_img_retriever.retrieve(img_tube), segmentor_cell.segments)

    #
    # Reshape image to multiple images fitted to the cell segments
    #
    shaper_cell.apply_to(img_prot, segmentor_cell.mask_segment).outline()
    percell_prot = shaper_cell.imgs_reshaped.copy()
    shaper_cell.apply_to(img_er, segmentor_cell.mask_segment).outline()
    percell_er = shaper_cell.imgs_reshaped.copy()

    for cell_counter in percell_prot.keys():
        viz.show_(percell_prot[cell_counter], percell_er[cell_counter])
    #viz.show_(*tuple(percell_prot.values()))

    print (df_labels.loc[cell_id])


    raise RuntimeError
