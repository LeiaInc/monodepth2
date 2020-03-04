import os
import PIL.Image as pil
import numpy as np

from .mono_dataset import MonoDataset

class HolopixDataset(MonoDataset):
    """
    Holopix Dataset which has left and right rectified image pairs.
    """
    def __init__(self, *args, **kwargs):
        super(HolopixDataset, self).__init__(*args, **kwargs)
        # These attributes not useful to Holopix Dataset
        self.K = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = None
        self.side_map = None

    def check_depth(self):
        """
        Always return False because Holopix dataset doesn't have ground truth depth
        """
        return False

    def get_image_path(self, folder, img_name):
        """Return the path to the image
        
        Arguments:
            folder {str} -- [the image_directory]
            img_name {str} -- [the image name]
        """
        return os.path.join(folder, img_name)

    def get_color(self, folder, img_name, do_flip):
        # The get_image_path function is modified to work with holopix dataset
        color = self.loader(self.get_image_path(folder, img_name))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color