from __future__ import print_function

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, TimeDistributed, Dense, noise


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
encoding_dim = 8

np.set_printoptions(threshold=np.nan)

print("Generating data...")

data = [[''.join(np.random.choice(list(chars))) for _ in range(9)] for _ in range(200000)]
for l in data:
    if l[0] == 'A':
        l.append('B')
    else:
        l.append('C')


X = np.zeros((len(data), 10, len(chars)), dtype=np.bool)

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=10)


test = [[''.join(np.random.choice(list(chars))) for _ in range(9)] for _ in range(2000)]
for l in test:
    if l[0] == 'A':
        l.append('B')
    else:
        l.append('C')

X_val = np.zeros((len(test), 10, len(chars)), dtype=np.bool)

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=10)

print("Creating model...")
model = Sequential()

model.add(noise.GaussianNoise(0.2, input_shape=(10, ACIDS)))

#Recurrent encoder
model.add(recurrent.LSTM(encoding_dim))
model.add(RepeatVector(10))

#And decoding
model.add(recurrent.LSTM(ACIDS, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy')

print("Let's go!")
# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, X, batch_size=128, nb_epoch=1,
              validation_data=(X_val, X_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        row = X_val[np.array([ind])]
        preds = model.predict_classes(row, verbose=0)
        correct = ctable.decode(row[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('T', correct)
        print('P', guess)
        print('---')

    beep = [''.join(np.random.choice(list(chars))) for _ in range(10)]
    row = np.zeros((len(test), 10, len(chars)), dtype=np.bool)
    row[0] = ctable.encode(beep, maxlen=10)
    preds = model.predict_classes(row, verbose=0)
    correct = ctable.decode(row[0])
    guess = ctable.decode(preds[0], calc_argmax=False)
    print('T', correct)
    print('P', guess)
    print('---')



