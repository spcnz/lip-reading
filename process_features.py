import os
import glob
import numpy as np
from vocabulary import Vocabulary

from constants import FEATURE_DIRECTORY_PATH, NUM_FACIAL_FEATURES, NUM_FRAMES, DEMO_FEATURE_DIRECTORY_PATH_SINGLE, DEMO_FEATURE_DIRECTORY_PATH_MULTY

DATASET_RANGE = list(range(1, 21)) + list(range(22, 35))

vocab = Vocabulary()

def condense_frames(frames, desired_length):

    # Already at or below desired length, nothing to be done.
    if len(frames) <= desired_length:
        return frames

    # We need to get one frame from every cond_ratio frames.
    cond_ratio = len(frames) * 1.0 / desired_length

    condensed_frames = np.zeros((desired_length, frames.shape[1]), dtype=frames.dtype)

    frame_count = 0.0

    for i in range(desired_length):
        idx = int(frame_count)
        end_idx = idx + 1
        weights = [idx + 1 - frame_count]

        # Frames that are contributing fully to this average have a weight of 1.
        while cond_ratio - sum(weights) >= 1.0:
            weights.append(1.0)
            end_idx += 1

        # Account for any remainder.
        if cond_ratio > sum(weights) + 0.0000001:
            weights.append(cond_ratio - sum(weights))
            end_idx += 1

        # Normalize the weights.
        norm = np.linalg.norm(weights)
        weights = [w / norm for w in weights]

        condensed_frames[i,:] = np.average(frames[idx:end_idx,:], axis=0, weights=weights)

        frame_count += cond_ratio

    return condensed_frames

def get_paths_test(speakers, is_single = True, is_demo = True):
    file_paths = []
    for sp in speakers:
        test_folder = 'test' if is_single and not is_demo else ''
        feature_path = FEATURE_DIRECTORY_PATH
        if (is_demo):
            if(is_single):
                feature_path = DEMO_FEATURE_DIRECTORY_PATH_SINGLE
            else:
                feature_path = DEMO_FEATURE_DIRECTORY_PATH_MULTY
        glob_data = os.path.join(feature_path, 's'+str(sp), test_folder, '*.npy')
        file_paths.extend(glob.glob(glob_data))       
    return file_paths

def get_paths_train(speakers, is_single = True):
    file_paths = []
    for sp in speakers:
        train_folder = 'train' if is_single else ''
        feature_path = FEATURE_DIRECTORY_PATH
        glob_data = os.path.join(feature_path, 's'+str(sp), train_folder, '*.npy')
        file_paths.extend(glob.glob(glob_data))       
    return file_paths


def files_to_features_and_labels(features_paths):
    features = []
    labels = []
    for path in features_paths:
        word = os.path.basename(path).split('.')[0].split('_')[2]
        feature = np.load(path)

        # ostecen video
        if feature.shape[0] == 0:
            continue

        # pauza u izgovaranju
        if word == 'sp':
            continue

        features.append(feature[:,:NUM_FACIAL_FEATURES])
        labels.append(word)

    return features, labels


def max_frames(features):
    max_frames = 0
    for f in features:
        if f.shape[0] > max_frames:
            max_frames = f.shape[0]
    return max_frames

def avg_frames(features):
    sum = 0
    for f in features:
        sum+=f.shape[0]
    return sum * 1.0/len(features)

def hot_encoding(words):
    global vocab
    hot_encoded_words = []
    for word in words:
        zeros = np.zeros(51, dtype=np.float32)
        zeros[vocab[word]] = 1.0
        hot_encoded_words.append(zeros)
    return hot_encoded_words



def get_features_train(speakers, is_single=True):
    global vocab

    feature_paths = get_paths_train(speakers, is_single)

    features, labels = load_features(feature_paths)
    
    return features, labels, vocab

def get_features_test(speakers, is_single = True, is_demo = True):
    global vocab
    
    feature_paths = get_paths_test(speakers, is_single = is_single, is_demo = is_demo)


    features, labels = load_features(feature_paths)
    
    return features, labels, vocab


def load_features(feature_paths, use_delta_frames=True):

    features, labels = files_to_features_and_labels(feature_paths)
    # recnik reci i matrica 
    labels = hot_encoding(labels)

    longest_word_max_frames = max_frames(features)

    if use_delta_frames:
        number_of_frames = NUM_FRAMES + 1
    else:
        number_of_frames = NUM_FRAMES

    for i, f in enumerate(features): 
        # POSLEDNJI FREJM - reshapeujem ga da ga stavim u matricu, to je poslednji frame u feature-u
        last_frame = np.array(f[-1].reshape(1, NUM_FACIAL_FEATURES))    # uzemi poslednji frejm

        # Add padding with duplicates of last frame.                    # dodavaj poslednji frejm dok ne dodjes do 6
        for _ in range(f.shape[0], number_of_frames):                  
            features[i] = np.concatenate((features[i], last_frame), axis=0)

        features[i] = condense_frames(features[i], number_of_frames)

        if use_delta_frames:
            # Take deltas.
            for j in range(1, len(features[i])):
                features[i][j-1,:] = features[i][j,:] - features[i][j-1,:]

            # Remove extra frame from the end.
            features[i] = features[i][:NUM_FRAMES, ...]

    # Convert to numpy arrays.
    features = np.asarray(features)
    labels = np.asarray(labels)


    return features, labels 






if __name__ == '__main__':
    load_features(DATASET_RANGE)