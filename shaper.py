'''Bla bla

'''
import numpy as np
import matplotlib.pyplot as plt

class ImageShapeMaker(object):
    '''Bla bla

    '''
    def __init__(self, img_data_accessor, reader_func, reader_func_kwargs={}):

        self.img_data_accessor = img_data_accessor
        self.reader_func = reader_func
        self.reader_func_kwargs = reader_func_kwargs

    def outline(self, img_path, mask):
        '''Bla bla

        '''
        img_prot = self.reader_func(img_path, **self.reader_func_kwargs)

        outlined_cell_imgs = {}
        for cell_counter, cell_mask in mask.items():
            img_single_cell = np.where(cell_mask, img_prot, -1)
            outlined_cell_imgs[cell_counter] = img_single_cell

            #fig, ax = plt.subplots(1,1)
            #ax.imshow(img_single_cell, cmap=plt.cm.jet)
            #plt.show()

        return outlined_cell_imgs

    def _rect(self, img_path, mask, post_func):
        '''Bla bla

        '''
        _rect_cell_imgs = {}
        for cell_counter, cell_outline in self.outline(img_path, mask).items():
            inside_mask_inds = np.argwhere(cell_outline > -1)

            x_min = min(inside_mask_inds[:,0])
            y_min = min(inside_mask_inds[:,1])
            x_max = max(inside_mask_inds[:,0])
            y_max = max(inside_mask_inds[:,1])

            _rect_cell_imgs[cell_counter] = post_func(cell_outline, x_min, x_max, y_min, y_max)

            fig, ax = plt.subplots(1,1)
            ax.imshow(_rect_cell_imgs[cell_counter], cmap=plt.cm.jet)
            plt.show()

        return _rect_cell_imgs

    def outline_rect(self, img_path, mask):
        '''Bla bla

        '''
        def _my_post_func(cell_outline, x_min, x_max, y_min, y_max):
            box_outline = np.full(cell_outline.shape, -1)
            box_outline[x_min: x_max + 1, y_min: y_max + 1] = 0
            return np.where(cell_outline < 0, box_outline, cell_outline)

        return self._rect(img_path, mask, _my_post_func)

    def cut_rect(self, img_path, mask):
        '''Bla bla

        '''
        def _my_post_func(cell_outline, x_min, x_max, y_min, y_max):
            box_cut = cell_outline[x_min:x_max + 1, y_min:y_max + 1]
            return np.where(box_cut < 0, 0, box_cut)

        return self._rect(img_path, mask, _my_post_func)




