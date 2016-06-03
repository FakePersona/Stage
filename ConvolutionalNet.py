from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, embeddings, Activation, convolutional, Dense, Flatten


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''

    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ctable = CharacterTable(chars, 10)

ACIDS = 26
encoding_dim = 30

print("Generating data...")

data = [[''.join(np.random.choice(['AB', 'CD', 'EF', 'GH', 'IJ', 'KL', 'MN', 'OP', 'QR', 'ST', 'UV', 'WX', 'YZ'])) for _ in range(5)] for _ in range(120000)] + [[''.join(np.random.choice(list(chars))) for _ in range(10)] for _ in range(80000)]

X = np.zeros((len(data), 10, len(chars)), dtype=np.bool)
Y = [[1] for _ in range(120000)] + [[0] for _ in range(80000)]

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=10)

test = [[''.join(np.random.choice((['AB', 'CD', 'EF', 'GH', 'IJ', 'KL', 'MN', 'OP', 'QR', 'ST', 'UV', 'WX', 'YZ']))) for _ in range(5)] for _ in range(1000)] + [[''.join(np.random.choice(list(chars))) for _ in range(10)] for _ in range(1000)]

X_val = np.zeros((len(test), 10, len(chars)), dtype=np.bool)
Y_val = [[1] for _ in range(1000)] + [[0] for _ in range(1000)]

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=10)

print("Creating model...")
Conv = Sequential()

Conv.add(convolutional.Convolution1D(10, 3, init='he_uniform', input_shape=(10, 26)))
Conv.add(Activation('relu'))

Conv.add(convolutional.Convolution1D(5, 3, init='he_uniform'))
Conv.add(Activation('relu'))

Conv.add(convolutional.Convolution1D(2, 3, init='he_uniform'))
Conv.add(Activation('relu'))

Conv.add(Flatten())
Conv.add(Dense(1, init='he_uniform'))
Conv.add(Activation('sigmoid'))

Conv.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])

print("Let's go!")
# Train the model each generation and show predictions against the validation datas
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    Conv.fit(X, Y, batch_size=128, nb_epoch=1,
              validation_data=(X_val, Y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        row = X_val[np.array([ind])]
        rowY = Y_val[np.array([ind])]
        preds = Conv.predict_classes(row, verbose=0)
        correct = rowY[0]
        guess = preds[0][0]
        print('T', correct)
        print('P', guess)
        print('---')

