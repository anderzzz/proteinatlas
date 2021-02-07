'''Image segmentor

'''
import numpy as np
from collections import Counter

from skimage.segmentation import watershed, random_walker
from skimage.draw import ellipse
from skimage import morphology
from skimage.measure import EllipseModel
from skimage.util import invert
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
        estimate_success = ellipse.estimate(coords)

        if not estimate_success:
            xc, yc = tuple([int(x) for x in np.median(coords, axis=0)])
            ellipse.params = xc, yc, 10, 10, 0.0

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


class _ConfocalWorker(object):
    '''Bla bla

    '''
    def __init__(self, img_retriever):
        self.img_retriever = img_retriever

        self.current_image = None

class _ConfocalAreaMasker(_ConfocalWorker):
    '''Bla bla

    '''
    def __init__(self, img_retriever,
                       edge_width=None,
                       body_luminosity=None, body_object_area=None, body_hole_area=None,
                       edge_luminosity=None, edge_object_area=None, edge_hole_area=None):

        super().__init__(img_retriever)
        self.edge_width = edge_width
        self.body_luminosity = body_luminosity
        self.body_object_area = body_object_area
        self.body_hole_area = body_hole_area
        self.edge_luminosity = edge_luminosity
        self.edge_object_area = edge_object_area
        self.edge_hole_area = edge_hole_area

        self.mask = None

    def denoise_and_thrs(self, img):
        '''Bla bla

        '''
        if self.edge_width is None:
            mask = self._denoise_thrs(img, self.body_luminosity, self.body_object_area, self.body_hole_area)

        else:
            mask_body = self._denoise_thrs(img, self.body_luminosity, self.body_object_area, self.body_hole_area)
            mask_edge = self._denoise_thrs(img, self.edge_luminosity, self.edge_object_area, self.edge_hole_area)

            body_mask = np.ones(img.shape, dtype=np.bool)
            body_mask[:self.edge_width + 1, :] = False
            body_mask[-self.edge_width - 1:, :] = False
            body_mask[:, :self.edge_width + 1] = False
            body_mask[:, -self.edge_width - 1:] = False
            mask = np.where(body_mask, mask_body, mask_edge)

        return mask

    def _denoise_thrs(self, img, thrs, max_object, max_hole):
        return morphology.remove_small_holes(
                   morphology.remove_small_objects(img > thrs, max_object), max_hole)


class _ConfocalMaskSegmentor(_ConfocalWorker):
    '''Bla bla

    '''
    def __init__(self, img_retriever):

        super().__init__(img_retriever)

        self.mask_segment = {}
        self.segments = None

    def cmp_mask_segments(self):

        for cell_counter in range(1, np.max(self.segments) + 1):
            self.mask_segment[cell_counter] = self.segments == cell_counter

    def del_segments(self, delete_list):
        '''Bla bla

        '''
        for cell_counter in delete_list:
            self.mask_segment.pop(cell_counter, None)

    def get_segments_on_edge(self, edge_thickness=1):
        '''Bla bla

        '''
        top_slice = self.segments[:edge_thickness, :]
        bottom_slice = self.segments[-edge_thickness:, :]
        left_slice = self.segments[:, :edge_thickness]
        right_slice = self.segments[:, -edge_thickness:]
        on_edge = np.union1d(np.unique(right_slice),
                             np.union1d(np.unique(left_slice),
                                        np.union1d(np.unique(top_slice),
                                                   np.unique(bottom_slice))))

        return [x for x in on_edge if x != 0]

    def fill_holes_in_masks(self):
        '''Bla bla

        '''
        new_mask_dict = {}
        for cell_counter, segment_mask in self.items():
            hole_free_segment_mask = ndimage.binary_fill_holes(segment_mask)
            new_mask_dict[cell_counter] = hole_free_segment_mask
        self.mask_segment = new_mask_dict

    @property
    def area_segments(self):
        cell_counter_counter = Counter(self.segments.flatten())
        cell_counter_counter['background'] = cell_counter_counter.pop(0)
        return cell_counter_counter

    @property
    def n_segments(self):
        return len(self.mask_segment)

    def __getitem__(self, item):
        return self.mask_segment[item]

    def items(self):
        for key, value in self.mask_segment.items():
            yield key, value


class ConfocalNucleusAreaMasker(_ConfocalAreaMasker):
    '''Bla bla
    
    '''
    def __init__(self, img_retriever,
                 edge_width=None,
                 body_luminosity=25, body_object_area=5000, body_hole_area=5000,
                 edge_luminosity=25, edge_object_area=6, edge_hole_area=5000):

        super().__init__(img_retriever,
                         edge_width,
                         body_luminosity, body_object_area, body_hole_area,
                         edge_luminosity, edge_object_area, edge_hole_area)
        
    def make_mask_(self, img_path):
        '''Bla bla
        
        '''
        img = self.img_retriever.retrieve(img_path)
        self.mask = self.denoise_and_thrs(img)
        

class ConfocalNucleusSegmentor(_ConfocalMaskSegmentor):
    '''Bla bla

    '''
    def __init__(self, img_retriever,
                 fit_ellipses=True):

        super().__init__(img_retriever)
        self.fit_ellipses = fit_ellipses

        self.segments_ellipse = {}

    def make_segments_(self, img_nuclei_path, nuclei_mask):
        '''Segment the plurality of nuclei in image, and fit ellipse

        This is done by (1) moderately denoise image by removing holes and objects, (2) identify each nucleus
        as any remaining contiguous set of bright points.

        '''
        img_nuclei = self.img_retriever.retrieve(img_nuclei_path)

        self.segments, _ = ndimage.label(nuclei_mask)
        self.cmp_mask_segments()
        if self.fit_ellipses:
            self.segments_ellipse = ellipse_fits(self.segments)

#        fig, ax = plt.subplots(1,1)
#        ax.imshow(img_nuclei, cmap='gray')
#        ax.imshow(self.segments, cmap=plt.cm.jet)
#        plt.show()

        return self

    def bounding_ellipse(self, bounding_n_neighbours=4, bounding_radial_slack=0.0):
        '''Construct bounding ellipse around each nucleus

        '''
        return bounding_ellipses(self.segments_ellipse, bounding_n_neighbours, bounding_radial_slack)


class ConfocalCellAreaMasker(_ConfocalAreaMasker):
    '''Bla bla
    
    '''
    def __init__(self, img_retriever,
                       body_luminosity=0, body_object_area=100, body_hole_area=100):
        
        super().__init__(img_retriever,
                         body_luminosity=body_luminosity, body_object_area=body_object_area, body_hole_area=body_hole_area)
        
    def make_mask_(self, img_path, nuclei_mask):
        '''Bla bla
        
        '''
        img = self.img_retriever.retrieve(img_path)
        img[nuclei_mask] = 255
        
        img_thrs = self.denoise_and_thrs(img)
        self.mask = morphology.binary_dilation(img_thrs, selem=morphology.disk(12))


class ConfocalCellSegmentor(_ConfocalMaskSegmentor):
    '''Bla bla

    '''
    def __init__(self, img_retriever):

        super().__init__(img_retriever)

    def make_segments_(self, img_path, cell_mask, nuclei_mask, nuclei_segments):

        img = self.img_retriever.retrieve(img_path)
        img[nuclei_mask] = 255
        self.current_image = img

        self.segments = watershed(invert(img), markers=nuclei_segments, mask=cell_mask, compactness=0)
        self.cmp_mask_segments()

        #for cell_counter, the_mask in self.items():
        #    print ('AA:{}'.format(cell_counter))
        #    fig, ax = plt.subplots(1,2)
        #    ax[0].imshow(self.current_image, cmap='gray')
        #    ax[0].imshow(the_mask, alpha=0.5, cmap=plt.cm.jet)
        #    ax[1].imshow(cell_mask)
        #    plt.show()

        return self



