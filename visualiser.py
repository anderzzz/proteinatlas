'''Bla bla

'''
import matplotlib.pyplot as plt
from skimage.color import label2rgb

class CellVisualiser(object):
    '''Visualise cell images

    '''
    def __init__(self, img_data_accessor, reader_func, reader_func_kwargs={}):

        self.img_data_accessor = img_data_accessor
        self.reader_func = reader_func
        self.reader_func_kwargs = reader_func_kwargs
        self.collection = None

    def set_collection(self, collection):
        self.collection = collection

    def channel(self, colour):
        if self.collection is None:
            raise RuntimeError('The image collection not set through `set_collection`')
        return self.reader_func(self.collection[colour], **self.reader_func_kwargs)

    def viz_cellmask_on_er(self, cellmask):
        '''Bla bla

        '''
        img_er = self.channel('ER')
