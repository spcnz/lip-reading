import sys
import numpy as np
import os

from keras.engine import Model
from tensorflow import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
import glob
from util import plot_stat
from constants import FEATURE_DIRECTORY_PATH, DEMO_FEATURE_DIRECTORY_PATH_SINGLE, DEMO_VIDEO_DIRECTORY_PATH_SINGLE, DEMO_FEATURE_DIRECTORY_PATH_MULTY, DEMO_VIDEO_DIRECTORY_PATH_MULTY
from process_features import get_features_test
from extract_features_vgg import extract_features

BATCH_SIZE_SINGLE = 32
BATCH_SIZE_MULTY = 70
# staviti parametar s1/2/3/4/5 da mu evaluira samo taj ili prazno za svih deset
# dodati feature extraction da se radi za single ce raditi samo feature extraction nad test delom tog speakera
# nad single svih 10 ce ucitati test za svakog, evaluira, isprinta, i na kraju mean(acc)
# nad multi ce ucitati 5 spikera komplet i za njih odraditi feature extraction
# plotovati u zavisnosti od parametra plot

SPEAKER_DEPENDENT_TEST = [1,2,3,4,5]
SPEAKER_INDEPENDENT_TEST = [30,31,32,33,34]
def get_parameters(test_type):
    if (test_type == 'single'):
        accuracies = []
        for s in SPEAKER_DEPENDENT_TEST:
            extract_features('s' + str(s), feature_path=DEMO_FEATURE_DIRECTORY_PATH_SINGLE, video_dir_path=DEMO_VIDEO_DIRECTORY_PATH_SINGLE, is_demo=True)
            
            features_test, labels_test, vocab = get_features_test(speakers=[s], is_single=True, is_demo=True)
            model = keras.models.load_model(f'model\\single_speaker\\model_from_speaker_{s}')
            _, acc = model.evaluate(features_test, labels_test, batch_size=BATCH_SIZE_SINGLE, verbose=False)

            accuracies.append(acc)
            print("Speaker : s", s)
            print('Accuracy: ', acc)
            print('Number of words for testing: ', features_test.shape[0])

            # correct = 0
            # for test_input, test_label in zip(features_test, labels_test):
            #     pred = model.predict(test_input.reshape(1, 6, 512))
            #     pred_idx = np.argsort(pred.reshape(len(vocab)))[::-1][0]
            #     corr_idx = np.argsort(test_label)[::-1][0]

            #     pred_word = vocab[pred_idx]
            #     corr_word = vocab[corr_idx]

            #     if pred_word == corr_word:
            #         correct += 1
            #     print (pred_word, '\t\t', corr_word)

        print(f"Mean accuracy: {np.sum(accuracies)/len(accuracies)}") 
    if (test_type == 'multy'):
        accuracies = []
        for s in SPEAKER_INDEPENDENT_TEST:
            extract_features('s' + str(s), feature_path=DEMO_FEATURE_DIRECTORY_PATH_MULTY, video_dir_path=DEMO_VIDEO_DIRECTORY_PATH_MULTY, is_demo=True)
    
            features_test, labels_test, vocab = get_features_test(speakers=[s], is_single=False, is_demo=True)
            model = keras.models.load_model('model\\20_speaker\\model')
            _, acc = model.evaluate(features_test, labels_test, batch_size=BATCH_SIZE_MULTY, verbose=False)

            accuracies.append(acc)
            print("Speaker : s", s)
            print('Accuracy: ', acc)
            print('Number of words for testing: ', features_test.shape[0])

            correct = 0
            print()
            print("PREDICTED\tTRUE")
            print("==========================")
            for test_input, test_label in zip(features_test, labels_test):
                pred = model.predict(test_input.reshape(1, 6, 512))
                pred_idx = np.argsort(pred.reshape(len(vocab)))[::-1][0]
                corr_idx = np.argsort(test_label)[::-1][0]

                pred_word = vocab[pred_idx]
                corr_word = vocab[corr_idx]

                if pred_word == corr_word:
                    correct += 1
                print (pred_word, '\t\t', corr_word)

        print(f"Mean accuracy: {np.sum(accuracies)/len(accuracies)}") 

def demo():
    if(len(sys.argv) > 1):
        print(sys.argv)
        get_parameters(sys.argv[1])
if __name__ == '__main__':
    demo()
    