import numpy as np

from keras.engine import Model
from tensorflow import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense


from process_features import get_features_test, get_features_train
from util import plot_stat

# Data
SPEAKERS = list(range(2, 15))
SPEAKERS = [1]

# Model
EPOCHS = 30 # 30 for single speaker
BATCH_SIZE = 48
ACTIVATION = 'tanh'
LOSS = 'mean_squared_error'
OPTIMIZER = 'adam'

# Load the data.
features_test, labels_test, vocab = get_features_test(speakers=SPEAKERS)

features_train, labels_train, vocab = get_features_train(speakers=SPEAKERS)


print(vocab.word_to_index_map)
print(len(vocab.word_to_index_map))


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

# Build model with layers
model = Sequential()
model.add(LSTM(128, input_shape=features_train[0].shape, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(vocab), activation=ACTIVATION))
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])

print(model.summary())

history = model.fit(features_train, labels_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
            verbose=True, validation_split=0.18)

_, acc = model.evaluate(features_test, labels_test, batch_size=BATCH_SIZE,
                        verbose=False)
print('Accuracy: ', acc)
plot_stat(history)

# model.save(f'model\\single_speaker\\modeel', include_optimizer=True,  overwrite=True)
model.save(f'model\\20_speaker\\model', include_optimizer=True,  overwrite=True)


# model = keras.models.load_model('model\\20_speaker\\model')

# features, labels, vocab = get_features_test(speakers=[10])

# _, acc = model.evaluate(features, labels, batch_size=BATCH_SIZE,
#                                  verbose=False)
# print('akkaka', acc)
