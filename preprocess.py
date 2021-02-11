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
from constants import FEATURE_DIRECTORY_PATH, VIDEO_DIRECTORY_PATH, ALIGN_DIRECTORY_PATH

def parse_alignment(file):
    f = open(file, 'r')
    lines = f.readlines()

    #segments will be list of tuples (start_frame_num, end_frame_num, word) for every word in alignment file
    segments = []

    #line in .align file is in format:   start_frame  end_frame  word
    for line in lines:

        align_elements = line.split()
        start_frame = align_elements[0]
        end_frame = align_elements[1]
        word = align_elements[2]

        # Round the start frame down and end frame up.
        start_frame_num = int(round(float(start_frame) / 1000))
        end_frame_num = int(round(float(end_frame) / 1000))

        segments.append((start_frame_num, end_frame_num, word))

    last_word_end_frame = segments[-1][1]
    firs_word_start_frame = segments[0][0]
    num_frames = last_word_end_frame - firs_word_start_frame
    print('num of frames per video', num_frames)

    # exclude first and last (cause it's sil)
    return segments[1:-1], num_frames

def process_video(speaker, vid_name):

    ''' Split a video into sets of frames corresponding to each spoken word. '''
    video_path = os.path.join(VIDEO_DIRECTORY_PATH, speaker, vid_name)
    align_name = vid_name.split('.')[0] + '.align'
    align_path = os.path.join(ALIGN_DIRECTORY_PATH, speaker, align_name)

    cap = cv2.VideoCapture(video_path)

    #alignments will be list of tuples (start_frame_num, end_frame_num, word) for every word in alignment file
    alignments, num_frames = parse_alignment(align_path)



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
    print(video_glob)
    video_paths = glob(video_glob)
    print('ovoo',video_paths)
    num_videos = len(video_paths)
    word_count = 0
    print(num_videos)

    stdscr.addstr(0, 0, f'Extracting facial features for speaker {speaker}. Press q to exit.')
    stdscr.refresh()

    try:
        for video_ordinal, video_path in enumerate(video_paths):
            video_name = os.path.basename(video_path)
            progress_msg(stdscr, video_ordinal, word_count, video_name, num_videos)

            process_video(speaker, video_name)
        
    except (SystemExit, KeyboardInterrupt):
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