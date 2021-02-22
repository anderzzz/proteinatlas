'''Reshape image given segmentation mask

'''
import numpy as np
import matplotlib.pyplot as plt

def check_after_apply(f):
    def wrapper(*args):
        if args[0].raw_image is None:
            raise RuntimeError('Before call to shaper methods, the `apply_to` method has to be called')
        return f(*args)
    return wrapper

class ImageShapeMaker(object):
    '''Bla bla

    '''
    def __init__(self, img_retriever):

        self.img_retriever = img_retriever

        self.raw_image = None
        self.imgs_reshaped = {}

    def apply_to(self, img_path, masks):
        '''Bla bla

        '''
        self.imgs_reshaped = {}
        self.raw_image = self.img_retriever.retrieve(img_path)
        for cell_counter, cell_mask in masks.items():
            img_single_cell = np.where(cell_mask, self.raw_image, -1)
            self.imgs_reshaped[cell_counter] = img_single_cell

        return self

    @check_after_apply
    def outline(self):
        '''Bla bla

        '''
        return self

    @check_after_apply
    def outline_rect(self):
        '''Bla bla

        '''
        def _my_post_func(cell_outline, x_min, x_max, y_min, y_max):
            box_outline = np.full(cell_outline.shape, -1)
            box_outline[x_min: x_max + 1, y_min: y_max + 1] = 0
            return np.where(cell_outline < 0, box_outline, cell_outline)

        return self._rect(_my_post_func)

    @check_after_apply
    def cut_rect(self):
        '''Bla bla

        '''
        def _my_post_func(cell_outline, x_min, x_max, y_min, y_max):
            box_cut = cell_outline[x_min:x_max + 1, y_min:y_max + 1]
            return np.where(box_cut < 0, 0, box_cut)

        return self._rect(_my_post_func)

    @check_after_apply
    def cut_square(self):
        '''Bla bla

        '''
        def _my_post_func(cell_outline, x_min, x_max, y_min, y_max):
            dx = x_max - x_min
            dy = y_max - y_min
            if dx >= dy:
                n_rows_add = dx - dy
                high = y_max
                low = y_min
                while n_rows_add > 0:
                    if high < self.raw_image.shape[1]:
                        high += 1
                        n_rows_add -= 1
                    if low > 0:
                        low -= 1
                        n_rows_add -= 1
                box_cut = cell_outline[x_min:x_max, low - n_rows_add:high]
            else:
                n_cols_add = dy - dx
                high = x_max
                low = x_min
                while n_cols_add > 0:
                    if high < self.raw_image.shape[0]:
                        high += 1
                        n_cols_add -= 1
                    if low > 0:
                        low -= 1
                        n_cols_add -= 1
                box_cut = cell_outline[low - n_cols_add:high, y_min:y_max]

            return np.where(box_cut < 0, 0, box_cut)

        return self._rect(_my_post_func)

    def _rect(self, post_func):
        '''Bla bla

        '''
        _rect_cell_imgs = {}
        for cell_counter, cell_outline in self.imgs_reshaped.items():
            inside_mask_inds = np.argwhere(cell_outline > -1)

            x_min = min(inside_mask_inds[:,0])
            y_min = min(inside_mask_inds[:,1])
            x_max = max(inside_mask_inds[:,0])
            y_max = max(inside_mask_inds[:,1])

            _rect_cell_imgs[cell_counter] = post_func(cell_outline, x_min, x_max, y_min, y_max)

            #fig, ax = plt.subplots(1,1)
            #ax.imshow(_rect_cell_imgs[cell_counter], cmap=plt.cm.jet)
            #plt.show()

        self.imgs_reshaped = _rect_cell_imgs

        return self

