#!/usr/bin/env python3
"""
script for metaseg input preparation
"""

import os
import numpy as np
from PIL import Image

from global_defs import CONFIG


class Cityscapes():  

    def __init__(self, **kwargs):
        """
        Dataset loader that processes all images from one specified root directory
        Also searches for images in every subdirectory in root directory
        """
        
        self.images = []    # where to load input images - absolute paths
        self.targets = []   # where to load ground truth if available - absolute paths
        self.preds = []     # where to load semantic segmentation predictions - absolute paths
        self.name = []      # image name

        for city in sorted(os.listdir(CONFIG.IMG_DIR)):
            for img in sorted(os.listdir(os.path.join(CONFIG.IMG_DIR,city))):

                self.images.append(os.path.join(CONFIG.IMG_DIR, city, img)) 
                self.targets.append(os.path.join(CONFIG.GT_DIR, city, img.replace('leftImg8bit','gtFine_labelTrainIds'))) 
                self.preds.append(os.path.join(CONFIG.PREDS_DIR, img.replace('.png','.npy')))  
                self.name.append(img.split('_left')[0])

    def __getitem__(self, index):
        """Generate one sample of data"""
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        target = np.asarray(Image.open(self.targets[index])) 
        return image, target, self.name[index], self.preds[index] 

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)

