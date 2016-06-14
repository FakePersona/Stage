from __future__ import print_function

import numpy as np

from keras import backend as K

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, TimeDistributed, Dense, Dropout, Convolution1D, Flatten

from Bio import SeqIO

def dist2(u,v):
    return sum([(x-y)**2 for (x, y) in zip(u, v)])

class AcidEmbedding(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen

        self.chars = list('rndeqkstchmavgilfpwybzuxXo')

        self.embed = [[1, 1, 9.09, 71.8],
                      [1, 0, 8.8, 2.4],
                      [1, -1, 9.6, 0.42],
                      [1, -1, 9.67, 0.72],
                      [1, 0, 9.13, 2.6],
                      [1, 1, 10.28, 200],
                      [1, 0, 9.15, 36.2],
                      [1, 0, 9.12, 200],
                      [0, 0, 10.78, 200],
                      [0, 1, 8.97, 4.19],
                      [0, 0, 9.21, 5.14],
                      [-1, 0, 9.87, 5.14],
                      [-1, 0, 9.72, 5.6],
                      [-1, 0, 9.6, 22.5],
                      [-1, 0, 9.76, 3.36],
                      [-1, 0, 9.6, 2.37],
                      [-1, 0, 9.24, 2.7],
                      [-1, 0, 10.6, 1.54],
                      [-1, 0, 9.39, 1.06],
                      [-1, 0, 9.11, 0.038],
                      [1, 0.5, 8.95, 37.1],
                      [1, -0.5, 9.40, 1.66],
                      [250, -250, 250, -250],
                      [0, 0, 0, 0],
                      [500, 500, 500, 500],
                      [-500, -500, -500, -500]]

        self.embed = [[x/500 for x in X] for X in self.embed]

        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, 4))
        for i, c in enumerate(C):
            X[i] = self.embed[self.char_indices[c]]
        return X


    def decode(self, X):
        prob = [[-dist2(x, y) for y in self.embed] for x in X]
        prob = (np.array(prob)).argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in prob)

chars = 'rndeqkstchmavgilfpwybzuxXo'
ctable = AcidEmbedding(150)

ACIDS = 4
encoding_dim = 500

np.set_printoptions(threshold=np.nan)

print("Generating data...")

data = []
test = []

record = SeqIO.parse("astral-scopedom-seqres-gd-sel-gs-bib-40-2.06.fa", "fasta")

for rec in record:
    if len(test) > 1999:
        break
    if len(rec.seq) > 149:
        continue
    if len(data) > 5999:
        test.append([rec.seq[i] for i in range(len(rec.seq))] + ['o' for _ in range(150 - len(rec.seq))])
    else:
        data.append([rec.seq[i] for i in range(len(rec.seq))] + ['o' for _ in range(150 - len(rec.seq))])

X = np.zeros((len(data), 150, ACIDS))

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=150)

X_val = np.zeros((len(test), 150, ACIDS))

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=150)

print("Creating model...")
model = Sequential()

#Convolutional encoder
model.add(Convolution1D(30, 5, activation='relu', input_shape=(150, ACIDS)))
model.add(Dropout(0.2))

model.add(Convolution1D(10, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(13, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(3, 2, activation='relu'))
model.add(Dropout(0.1))

model.add(Convolution1D(6, 2, activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(encoding_dim))

model.add(RepeatVector(150))

#And decoding
model.add(recurrent.LSTM(ACIDS, return_sequences=True))

model.load_weights("20prot_pad_emb.h5")

model.compile(optimizer='rmsprop', loss='mean_squared_error')

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
        preds = model.predict(row, verbose=0)
        correct = ctable.decode(row[0])
        guess = ctable.decode(preds[0])
        print('T', correct)
        print('P', guess)
        print('---')

model.save_weights("20prot_pad_emb.h5", overwrite=True)
