'''Bla bla

'''
import numpy as np
import matplotlib.pyplot as plt

def check_after_apply(f):
    def wrapper(*args):
        if args[0].image is None:
            raise RuntimeError('Before call to shaper methods, the `apply_to` method has to be called')
        return f(*args)
    return wrapper

class ImageShapeMaker(object):
    '''Bla bla

    '''
    def __init__(self, img_data_accessor, reader_func, reader_func_kwargs={}):

        self.img_data_accessor = img_data_accessor
        self.reader_func = reader_func
        self.reader_func_kwargs = reader_func_kwargs

        self.image = None
        self.masks = None
        self.imgs_shaped = {}

    def apply_to(self, img_path, masks):
        '''Bla bla

        '''
        self.image = self.reader_func(img_path, **self.reader_func_kwargs)
        self.masks = masks

        return self

    @check_after_apply
    def outline(self):
        '''Bla bla

        '''
        for cell_counter, cell_mask in self.masks.items():
            img_single_cell = np.where(cell_mask, self.image, -1)
            self.imgs_shaped[cell_counter] = img_single_cell

            #fig, ax = plt.subplots(1,1)
            #ax.imshow(img_single_cell, cmap=plt.cm.jet)
            #plt.show()

        return self

    def _rect(self, post_func):
        '''Bla bla

        '''
        _rect_cell_imgs = {}
        for cell_counter, cell_outline in self.imgs_shaped.items():
            inside_mask_inds = np.argwhere(cell_outline > -1)

            x_min = min(inside_mask_inds[:,0])
            y_min = min(inside_mask_inds[:,1])
            x_max = max(inside_mask_inds[:,0])
            y_max = max(inside_mask_inds[:,1])

            _rect_cell_imgs[cell_counter] = post_func(cell_outline, x_min, x_max, y_min, y_max)

            fig, ax = plt.subplots(1,1)
            ax.imshow(_rect_cell_imgs[cell_counter], cmap=plt.cm.jet)
            plt.show()

        self.imgs_shaped = _rect_cell_imgs

        return self

    @check_after_apply
    def outline_rect(self):
        '''Bla bla

        '''
        def _my_post_func(cell_outline, x_min, x_max, y_min, y_max):
            box_outline = np.full(cell_outline.shape, -1)
            box_outline[x_min: x_max + 1, y_min: y_max + 1] = 0
            return np.where(cell_outline < 0, box_outline, cell_outline)

        self.outline()
        return self._rect(_my_post_func)

    @check_after_apply
    def cut_rect(self):
        '''Bla bla

        '''
        def _my_post_func(cell_outline, x_min, x_max, y_min, y_max):
            box_cut = cell_outline[x_min:x_max + 1, y_min:y_max + 1]
            return np.where(box_cut < 0, 0, box_cut)

        self.outline()
        return self._rect(_my_post_func)





