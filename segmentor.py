'''Image segmentor

'''
import numpy as np

from skimage.segmentation import watershed
from skimage.draw import ellipse
from skimage import morphology
from skimage.measure import EllipseModel
from scipy import ndimage

import matplotlib.pyplot as plt

def ellipse_fits(img_labelled):
    '''Bla bla

    '''
    ellipses_by_label = {}
    for k_label in range(1, img_labelled.max() + 1):
        rows, cols = np.where(img_labelled == k_label)
        coords = np.stack([rows, cols]).T
        ellipse = EllipseModel()
        ellipse.estimate(coords)

        ellipses_by_label[k_label] = ellipse

    return ellipses_by_label

def bounding_ellipses(ellipses_nuclei, n_neighbour_nuclei=1, radial_slack=0.0):
    '''Bla bla

    '''
    ellipses_bounding = {}
    for k_label, ellipse_n in ellipses_nuclei.items():
        x0, y0, a0, b0, theta0 = ellipse_n.params
        dists = []
        for l_label, ellipse_n_test in ellipses_nuclei.items():
            xt, yt, at, bt, thetat = ellipse_n_test.params
            dist = (((xt - x0) * np.cos(theta0) + (yt - y0) * np.sin(theta0)) / a0) ** 2 + \
                   (((xt - x0) * np.sin(theta0) - (yt - y0) * np.cos(theta0)) / b0) ** 2
            dists.append((l_label, dist))

        sort_dist = sorted(dists, key=lambda x: x[1])
        scale = np.sqrt(sort_dist[n_neighbour_nuclei][1]) * (1.0 + radial_slack)

        ellipse_bounder = EllipseModel()
        ellipse_bounder.params = (round(x0), round(y0), round(a0 * scale), round(b0 * scale), -theta0)

        ellipses_bounding[k_label] = ellipse_bounder

    return ellipses_bounding

class _ConfocalSegmentor(object):
    '''Bla bla

    '''
    def __init__(self, img_data_accessor, reader_func, reader_func_kwargs={}):
        self.img_data_accessor = img_data_accessor
        self.reader_func = reader_func
        self.reader_func_kwargs = reader_func_kwargs

    def _denoise_and_thrs(self, img, thrs_value, max_area_object, max_area_hole):
        return morphology.remove_small_holes(
                   morphology.remove_small_objects(img > thrs_value, max_area_object), max_area_hole)

class ConfocalNucleusSegmentor(_ConfocalSegmentor):
    '''Bla bla

    '''
    def __init__(self, img_data_accessor, reader_func, reader_func_kwargs={},
                 denoise_upper_luminosity=25, denoise_object_area=5000, denoise_hole_area=5000):

        super().__init__(img_data_accessor, reader_func, reader_func_kwargs)
        self.denoise_upper_luminosity = denoise_upper_luminosity
        self.denoise_object_area = denoise_object_area
        self.denoise_hole_area = denoise_hole_area

        self.mask = None
        self.segments_id = None
        self.n_segments = None
        self.segments_ellipse = None

    def segment(self, img_nuclei_path):
        '''Segment the plurality of nuclei in image, and fit ellipse

        This is done by (1) moderately denoise image by removing holes and objects, (2) identify each nucleus
        as any remaining contiguous set of bright points.

        '''
        img_nuclei = self.retrieve_image(img_nuclei_path)

        self.mask = self._denoise_and_thrs(img_nuclei, self.denoise_upper_luminosity,
                                           self.denoise_object_area, self.denoise_hole_area)
        #self.mask = np.invert(mask)
        self.segments_id, self.n_segments = ndimage.label(self.mask)
        self.segments_ellipse = ellipse_fits(self.segments_id)

    def bounding_ellipse(self, bounding_n_neighbours=4, bounding_radial_slack=0.0):
        '''Construct bounding ellipse around each nucleus

        '''
        return bounding_ellipses(self.segments_ellipse, bounding_n_neighbours, bounding_radial_slack)

    def retrieve_image(self, img_nuclei_path):
        '''Bla bla

        '''
        return self.reader_func(img_nuclei_path, **self.reader_func_kwargs)

class ConfocalCellSegmentor(_ConfocalSegmentor):
    '''Bla bla

    '''
    def __init__(self, img_data_accessor, reader_func, reader_func_kwargs={},
                 nucleus_segmentor=None,
                 denoise_lower_luminosity=50, denoise_object_area=16, denoise_hole_area=16):

        super().__init__(img_data_accessor, reader_func, reader_func_kwargs)
        self.nucleus_segmentor = nucleus_segmentor
        self.denoise_lower_luminosity = denoise_lower_luminosity
        self.denoise_object_area = denoise_object_area
        self.denoise_hole_area = denoise_hole_area

        self.mask = {}

    def segment(self, img_er_path, img_nuclei_path):

        # Segment nucleus
        self.nucleus_segmentor.segment(img_nuclei_path)
        nucleus_bounding_ellipses = self.nucleus_segmentor.bounding_ellipse()

        # Add nuclei at maximum intensity to ER image
        img_cell = self.reader_func(img_er_path, **self.reader_func_kwargs)
        img_cell[self.nucleus_segmentor.mask] = 255
        height, width = img_cell.shape

        # Threshold the image
        img_thrs = self._denoise_and_thrs(img_cell, self.denoise_lower_luminosity,
                                          self.denoise_object_area, self.denoise_hole_area)

        # Run segmentation on masked image data
        for cell_id, ellipse_b in nucleus_bounding_ellipses.items():

            # Mask for content within bounding ellipse (and within frame of image)
            ellipse_kwargs = dict(zip(('r', 'c', 'r_radius', 'c_radius', 'rotation'), ellipse_b.params))
            ellipse_kwargs['rotation'] = ellipse_kwargs['rotation'] * -1.0
            rr, cc = ellipse(**ellipse_kwargs)

            mask_inside_frame = ((rr >= 0) & (rr < height)) & ((cc >= 0) & (cc < width))
            rr = rr[mask_inside_frame]
            cc = cc[mask_inside_frame]

            mask_cell_area = np.zeros(img_cell.shape, dtype=np.bool)
            mask_cell_area[rr, cc] = True

            # Initialize cell labels, one unique value per nucleus within bounding ellipse mask
            labels = np.copy(self.nucleus_segmentor.segments_id)
            labels[~mask_cell_area] = -1

            segmented = watershed(~img_thrs, labels)
            self.mask[cell_id] = segmented == labels[ellipse_kwargs['r'], ellipse_kwargs['c']]


        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(img_cell, cmap='gray')
        ax[0,0].imshow(self.mask[5], alpha=0.5, cmap=plt.cm.jet)
        ax[0,1].imshow(img_cell, cmap='gray')
        ax[0,1].imshow(self.mask[9], alpha=0.5, cmap=plt.cm.jet)
        ax[1,0].imshow(img_cell, cmap='gray')
        ax[1,0].imshow(self.mask[14], alpha=0.5, cmap=plt.cm.jet)
        ax[1,1].imshow(img_cell, cmap='gray')
        ax[1,1].imshow(self.mask[15], alpha=0.5, cmap=plt.cm.jet)
        plt.show()

        fig, ax = plt.subplots(1,1)
        div_mask = np.zeros(img_cell.shape)
        for cell_id, mask in self.mask.items():
            div_mask[mask] = cell_id
        ax.imshow(img_cell, cmap='gray')
        ax.imshow(div_mask, alpha=0.5, cmap=plt.cm.jet)
        plt.show()