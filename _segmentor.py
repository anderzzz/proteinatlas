'''Image segmentor

'''
import numpy as np
from collections import Counter

from skimage.segmentation import watershed, random_walker
from skimage import morphology
from skimage.measure import EllipseModel
from skimage.util import invert
from scipy import ndimage

import matplotlib.pyplot as plt

class ConfocalAbsentCellCounterError(Exception):
    pass

def ellipse_fits(img_labelled, exclusion_list=[0]):
    '''Bla bla

    '''
    ellipses_by_label = {}
    for k_label in np.unique(img_labelled):
        if k_label in exclusion_list:
            continue

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

        self.mask_segment = {}
        for cell_counter in np.unique(self.segments):
            if cell_counter == 0:
                continue
            self.mask_segment[cell_counter] = self.segments == cell_counter

    def del_segment(self, id, error_if_absent=False):
        '''Bla bla

        '''
        removed = self.mask_segment.pop(id, None)
        if error_if_absent and removed is None:
            raise ConfocalAbsentCellCounterError('Cell id {} absent so cannot be deleted'.format(id))
        self.segments[self.segments == id] = 0

    def get_segments_on_edge(self, edge_thickness=1, exclude_null=True):
        '''Bla bla

        '''
        edge_pixels = self.get_edge_pixels(edge_thickness)
        on_edge = np.unique(edge_pixels)
        if exclude_null:
            on_edge = [x for x in on_edge if x != 0]

        return on_edge

    def get_segments_areafraction_on_edge(self, edge_thickness):
        '''Bla bla

        '''
        edge_areas = Counter(self.get_edge_pixels(edge_thickness))
        edge_areas.pop(0, None)
        total_areas = self.get_area_segments()

        fraction_on_edge = {}
        for cell_counter in edge_areas:
            fraction_on_edge[cell_counter] = edge_areas[cell_counter] / total_areas[cell_counter]

        return fraction_on_edge

    def get_edge_pixels(self, edge_thickness):
        '''Bla bla

        '''
        top_slice = self.segments[:edge_thickness, :]
        bottom_slice = self.segments[-edge_thickness:, :]
        left_slice = self.segments[edge_thickness:-edge_thickness, :edge_thickness]
        right_slice = self.segments[edge_thickness:-edge_thickness, -edge_thickness:]

        return np.concatenate([top_slice.flatten(), bottom_slice.flatten(),
                               left_slice.flatten(), right_slice.flatten()])

    def get_area_segments(self):
        '''Bla bla

        '''
        cell_counter_counter = Counter(self.segments.flatten())
        cell_counter_counter['background'] = cell_counter_counter.pop(0)
        return cell_counter_counter

    def fill_holes(self):
        '''Bla bla

        '''
        new_mask_dict = {}
        for cell_counter, segment_mask in self.items():
            hole_free_segment_mask = ndimage.binary_fill_holes(segment_mask)
            new_mask_dict[cell_counter] = hole_free_segment_mask

            self.segments[hole_free_segment_mask] = cell_counter

        self.mask_segment = new_mask_dict

    @property
    def n_segments(self):
        return len(self.mask_segment)

    def __getitem__(self, item):
        return self.mask_segment[item]

    def keys(self):
        return self.mask_segment.keys()

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
                       body_luminosity=0, body_object_area=100, body_hole_area=100,
                       fuzzy_boundary=5):
        
        super().__init__(img_retriever,
                         body_luminosity=body_luminosity, body_object_area=body_object_area, body_hole_area=body_hole_area)
        self.fuzzy_boundary = fuzzy_boundary
        
    def make_mask_(self, img_path, nuclei_mask):
        '''Bla bla
        
        '''
        img = self.img_retriever.retrieve(img_path)
        img[nuclei_mask] = 255
        
        img_thrs = self.denoise_and_thrs(img)
        self.mask = morphology.binary_dilation(img_thrs, selem=morphology.disk(self.fuzzy_boundary))


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


class ConfocalNucleusSweepAreaMasker(object):
    '''Bla bla

    '''
    def __init__(self, img_retriever, maskers_sweep=[]):

        self.img_retriever = img_retriever
        self.maskers_sweep = maskers_sweep
#        self.mask_sweeps = []
        self.mask = None

    def make_mask_sweep_(self, img):
        '''Bla bla

        '''
        for masker in self.maskers_sweep:
            masker.make_mask_(img)
#            self.mask_sweeps.append(masker.mask)

    def infer_mask_from_segments_(self, segments):
        '''Bla bla

        '''
        self.mask = np.where(segments != 0, True, False)


class ConfocalNucleusSweepSegmentor(ConfocalNucleusSegmentor):
    '''Bla bla

    '''
    def __init__(self, img_retriever, fit_ellipses=True):

        super().__init__(img_retriever, fit_ellipses=False)
        self.fit_ellipses = fit_ellipses

    def make_segments_(self, img_path, nuclei_mask):
        '''Bla bla

        '''
        segments_sweeps = []
        for masker in nuclei_mask:
            super(ConfocalNucleusSweepSegmentor, self).make_segments_(img_path, masker.mask)
            segments_sweeps.append(self.segments.copy())

#            fig, ax = plt.subplots(1,1)
#            ax.imshow(self.img_retriever.retrieve(img_path), cmap='gray')
#            ax.imshow(segments_sweeps[-1], alpha=0.5)
#            plt.show()

        self.segments = self._merge_sweeps(segments_sweeps, img_path)
        self.cmp_mask_segments()
        if self.fit_ellipses:
            self.segments_ellipse = ellipse_fits(self.segments)

        return self

    def _merge_sweeps(self, segments_sweeps, img_path):
        '''Bla bla

        '''
        if len(segments_sweeps) == 1:
            keys = np.unique(segments_sweeps[0])
            vals = np.array(range(len(keys)))
            mapping_ar = np.zeros(keys.max() + 1, dtype=int)
            mapping_ar[keys] = vals

            return mapping_ar[segments_sweeps[0]]

        segments_conformed = segments_sweeps.pop(0)

        lower_bg_thrs_segments = segments_sweeps[0]
        new_segment_id = max(lower_bg_thrs_segments.max(), segments_conformed.max())
        for segment_counter in range(1, lower_bg_thrs_segments.max() + 1):
            index_segment_lower = np.argwhere(lower_bg_thrs_segments == segment_counter)
            segments_conformed_ids = np.unique(segments_conformed[index_segment_lower[:,0], index_segment_lower[:,1]])
            nonzero_ids = [x for x in segments_conformed_ids if x!= 0]

            if len(nonzero_ids) == 0:
                new_segment_id += 1
                segments_conformed = np.where(lower_bg_thrs_segments == segment_counter, new_segment_id, segments_conformed)

            elif len(nonzero_ids) == 1:
                segments_conformed = np.where(lower_bg_thrs_segments == segment_counter, nonzero_ids[0], segments_conformed)

        segments_sweeps_next = [segments_conformed] + segments_sweeps[1:]
#        fig, ax = plt.subplots(1,1)
#        ax.imshow(self.img_retriever.retrieve(img_path), cmap='gray')
#        ax.imshow(segments_sweeps_next[0], cmap='jet', alpha=0.5)
#        plt.show()

        return self._merge_sweeps(segments_sweeps_next, img_path)


