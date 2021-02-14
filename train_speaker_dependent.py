import numpy as np

from keras.engine import Model
from tensorflow import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
import json 

from process_features import get_features_test, get_features_train
from util import plot_stat
# trenirati sa i bez delte, plotovati i grupisati plotove i accuracie u foldere 
# Mean accuracy: 0.785445362329483
# Mean accuracy without normalization 0.8228423655033111
# Data
SPEAKERS = [1,2,3,4,5,6,7,8,9,10]
# Model
EPOCHS = 30 # 30 for single speaker
BATCH_SIZE = 32
ACTIVATION = 'tanh'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'

# Load the data.

accuracies = []
for speaker in SPEAKERS:
    features_test, labels_test, vocab = get_features_test(speakers=[speaker], is_single=True, is_demo=False)
    features_train, labels_train, vocab = get_features_train(speakers=[speaker], is_single=True) 
    model = Sequential()
    model.add(LSTM(128, input_shape=features_train[0].shape, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(len(vocab), activation=ACTIVATION))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
    print(f"Current Speaker Training: {speaker}")
    print(model.summary())

    history = model.fit(features_train, labels_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                verbose=True, validation_split=0.2) # 0.18 znaci da je ukupan dataset splitovan u ondnosu 70/15/15

    _, acc = model.evaluate(features_test, labels_test, batch_size=BATCH_SIZE,
                            verbose=False)
    accuracies.append(acc)
    print(f'Accuracy for speaker {speaker}: ', acc)

    model.save(f'model\\single_speaker\\model_from_speaker_{speaker}', include_optimizer=True, overwrite=True)

    print('TRAIN FEATURES LEN', len(features_train))
    print('TRAIN LABELS LEN', len(labels_train))

    print('TEST FEATURES LEN', len(features_test))
    print('TEST LABELS LEN', len(labels_test))

    frames_per_word = features_train.shape[1]
    features_per_frame = features_train.shape[2]

    print('Number of words for training: ', features_train.shape[0])
    print('Number of words for testing: ', features_test.shape[0])
    print('Frames per word: ', frames_per_word)
    print('Features per frame: ', features_per_frame)



    # daa = keras.models.load_model('model\\single_speaker\\model_from_speaker_1')
    # _, acc = daa.evaluate(features_test, labels_test, batch_size=BATCH_SIZE, verbose=False)

    print('Accuracy: ', acc)    
print(f"Mean accuracy: {np.sum(accuracies)/len(accuracies)}")




    #plot_stat(history)
# model = keras.models.load_model('model\\20_speaker\\model')

# features, labels, vocab = get_features_test(speakers=[10])

# _, acc = model.evaluate(features, labels, batch_size=BATCH_SIZE,
#                                  verbose=False)
# print('akkaka', acc)
