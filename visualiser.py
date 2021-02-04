'''Bla bla

'''
import numpy as np
import matplotlib.pyplot as plt

class Visualiser(object):

    def __init__(self, intensity_range=(0, 255), cmap='gray'):
        self.intensity_range = intensity_range
        self.cmap = cmap

    def show_(self, *args):

        n_imgs = len(args)
        nrow = int(np.floor(np.sqrt(n_imgs)))
        ncol = int(np.ceil(n_imgs / nrow))
        fig, ax = plt.subplots(nrow, ncol)
        for k_img, img in enumerate(args):
            k_row = k_img // ncol
            k_col = k_img % ncol
            ax[k_row, k_col].imshow(img, cmap=self.cmap, vmin=self.intensity_range[0], vmax=self.intensity_range[1])

        plt.show()

