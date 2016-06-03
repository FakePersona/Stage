from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, Dense


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
        X = np.zeros(maxlen)
        for i, c in enumerate(C):
            X[i] = self.char_indices[c]
        return X

    def decode(self, X, calc_argmax=True):
        return ''.join(self.indices_char[x] for x in X)

chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ctable = CharacterTable(chars, 20)

ACIDS = 26
encoding_dim = 10

print("Generating data...")
data = [list('ABCDEFGHIJKLMNO') + [''.join(np.random.choice(list(chars))) for _ in range(5)] for _ in range(200000)]

X = np.zeros((len(data), 20), dtype=np.bool)

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=20)


test = [list('ABCDEFGHIJKLMNO') + [''.join(np.random.choice(list(chars))) for _ in range(5)] for _ in range(2000)]

X_val = np.zeros((len(test), 20), dtype=np.bool)

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=20)

print("Creating model...")

model = Sequential()

model.add(Dense(encoding_dim, input_shape=(20,)))
model.add(Activation('relu'))

model.add(Dense(20))

model.compile(optimizer='sgd', loss='binary_crossentropy')


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


