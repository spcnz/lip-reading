from __future__ import print_function

import os
import cv2
import numpy as np
np.random.seed(1337)
import sys

from keras.layers import Input
from keras_vggface.vggface import VGGFace

from vocab import Vocabulary

def preprocess(speaker):

    # input_tensor = Input(shape=(224, 224, 3))
    # vgg_model = VGGFace(input_tensor=input_tensor, include_top=False, pooling='avg')
    pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: Must specify at least one speaker.')
        print('python preprocess.py sp1 [sp2] ...')
    else:
        for speaker in sys.argv[1:]:
            preprocess(speaker)