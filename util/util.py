from __future__ import print_function
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import os


class util():
    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def gkern_2d(self, size=5, sigma=3):
        # Create 2D gaussian kernel
        dirac = np.zeros((size, size))
        dirac[size // 2, size // 2] = 1
        mask = gaussian_filter(dirac, sigma)
        # Adjust dimensions for torch conv2d
        return np.stack([np.expand_dims(mask, axis=0)] * 3)

    def mkdirs(self, paths):
        if isinstance(paths, list) and not isinstance(paths, str):
            for path in paths:
                self.mkdir(path)
        else:
            self.mkdir(paths)

    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
