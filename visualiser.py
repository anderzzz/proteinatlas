'''Bla bla

'''
import copy
import numpy as np
import matplotlib.pyplot as plt

class Visualiser(object):

    def __init__(self, intensity_range=(0, 255), cmap='jet', cmap_set_under='black', cmap_set_over='white'):
        self.intensity_range = intensity_range
        self.cmap = copy.copy(plt.cm.get_cmap(cmap))
        if not cmap_set_under is None:
            self.cmap.set_under(cmap_set_under)
        if not cmap_set_over is None:
            self.cmap.set_over(cmap_set_over)

    def show_(self, *args):

        n_imgs = len(args)
        nrow = int(np.floor(np.sqrt(n_imgs)))
        ncol = int(np.ceil(n_imgs / nrow))
        fig, ax = plt.subplots(nrow, ncol)
        for k_img, img in enumerate(args):
            if nrow > 1:
                k_row = k_img // ncol
                k_col = k_img % ncol
                ax[k_row, k_col].imshow(img, cmap=self.cmap, vmin=self.intensity_range[0], vmax=self.intensity_range[1])
            else:
                ax[k_img].imshow(img, cmap=self.cmap, vmin=self.intensity_range[0], vmax=self.intensity_range[1])

        plt.show()

    def show_segments_overlay(self, background_img, segments, alpha=0.5):

        fig, ax = plt.subplots(1,1)
        ax.imshow(background_img, cmap='gray')
        ax.imshow(segments, alpha=alpha)
        plt.show()

