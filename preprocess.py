from __future__ import print_function

import cv2
import numpy as np
from glob import glob
np.random.seed(1337)
import sys

from keras.layers import Input
from keras_vggface.vggface import VGGFace
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from vocabulary import Vocabulary
from util import curses_init, curses_clean_up, progress_msg


VIDEO_DIRECTORY_PATH = 'data/videos'
ALIGN_DIRECTORY_PATH = 'data/align'
FEATURE_DIRECTORY_PATH = 'data/features'

DEFAULT_TRAINING_FRACTION = 0.7

# Number of frames to use for each word.
NUM_FRAMES = 6

# Number of facial features to use. Maximum is 512.
NUM_FACIAL_FEATURES = 512


def preprocess(speaker):

    stdscr = curses_init()

    input_tensor = Input(shape=(224, 224, 3))

    #don't include 3 fully connected layers on top
    vgg_model = VGGFace(input_tensor=input_tensor, include_top=False, pooling='avg')

    #create folder for speaker's features
    speaker_feature_dir = os.path.join(FEATURE_DIRECTORY_PATH, speaker)
    if not os.path.isdir(speaker_feature_dir):
        os.mkdir(speaker_feature_dir)

    # Get list of all videos to process.
    video_glob = os.path.join(VIDEO_DIRECTORY_PATH, speaker, '*.mpg')
    video_paths = glob(video_glob)
    num_videos = len(video_paths)
    word_count = 0

    stdscr.addstr(0, 0, f'Extracting facial features for speaker {speaker}. Press q to exit.')
    stdscr.refresh()

    try:
        for video_ordinal, video_path in enumerate(video_paths):
            video_name = os.path.basename(video_path)
            progress_msg(stdscr, video_count, word_count, video_name, num_videos)

    except (SystemExit, KeyboardInterrupt):
        # Expected exceptions are either the generic one raised when a user
        # presses 'q' to exit, or a KeyboardInterrupt if the user gets
        # impatient and presses Ctrl-C.
        pass
    finally:
        curses_clean_up()




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: Must specify at least one speaker.')
        print('python preprocess.py sp1 [sp2] ...')
    else:
        for speaker in sys.argv[1:]:
            preprocess(speaker)