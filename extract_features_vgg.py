import cv2
import numpy as np
from glob import glob
np.random.seed(1337)
import sys

from keras.layers import Input
from keras_vggface.vggface import VGGFace
import os
import math

from vocabulary import Vocabulary
from util import curses_init, curses_clean_up, progress_msg
from constants import FEATURE_DIRECTORY_PATH, VIDEO_DIRECTORY_PATH, ALIGN_DIRECTORY_PATH

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
        start_frame_num = int(math.floor(float(start_frame) / 1000))
        end_frame_num = int(math.ceil(float(end_frame) / 1000))

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

    #alignments will be list of tuples (start_frame, end_frame, word) for every word in alignment file 
    alignments, num_frames = parse_alignment(align_path)
    frames = np.ndarray(shape=(num_frames, 224, 224, 3), dtype=np.float32)
    frame_num = 0
    has_frame, img = cap.read()

    while has_frame:
        resized_frame = cv2.resize(img, (224, 224)).astype(np.float32)

        resized_frame = np.expand_dims(resized_frame, axis=0)
        # Zero-center by mean pixel
        resized_frame[:, :, :, 0] -= 93.5940
        resized_frame[:, :, :, 1] -= 104.7624
        resized_frame[:, :, :, 2] -= 129.1863

        #add resized_frame into frames array
        frames[frame_num,:,:,:] = resized_frame

        has_frame, img = cap.read()

        frame_num += 1

    # Divide up frames based on mapping to spoken words.
    word_frames = []
    words_arr = []
    for seg in alignments:
        word_frames.append(frames[seg[0]:seg[1],:,:])
        words_arr.append(seg[2])

    return word_frames, words_arr

def extract_features(speaker):

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
   
    stdscr.addstr(0, 0, f'Extracting facial features for speaker {speaker}.')
    stdscr.refresh()

    try:
        for video_ordinal, video_path in enumerate(video_paths):
            video_name = os.path.basename(video_path)

            progress_msg(stdscr, video_ordinal, word_count, video_name, num_videos)

            word_frames, words_arr = process_video(speaker, video_name)

            name_no_ext = video_name.split('.')[0]
            
            # Process each word's set of frames.
            for i, word_frame in enumerate(word_frames):

                word_count += 1
                progress_msg(stdscr, video_ordinal, word_count, video_name, num_videos)

                # Format of the file name is [video_name]_[word_index]_[word].
                feature_file_name = f'{name_no_ext}_{i}_{words_arr[i]}'
                
                
                feature_file_path = os.path.join(speaker_feature_dir, feature_file_name)
            
                if (feature_exists(feature_file_path)):
                    continue
 
                # Classify the frames and save the features to a file.
                features = vgg_model.predict(word_frame)
                np.save(feature_file_path, features)
    except (KeyboardInterrupt):
        pass
    finally:
        curses_clean_up()


def feature_exists(feature_file_path):
    '''checks if feature is already generated'''

    return os.path.isfile(feature_file_path + '.npy')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: Must specify at least one speaker.')
        print('python preprocess.py sp1 [sp2] ...')
    else:
        for speaker in sys.argv[1:]:
            extract_features(speaker)