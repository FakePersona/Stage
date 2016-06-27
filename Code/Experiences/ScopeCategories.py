from __future__ import print_function

import numpy as np

from keras import backend as K

from sklearn import cluster

from Bio import SeqIO

from keras.models import Sequential
from keras.layers import recurrent, RepeatVector, Activation, TimeDistributed, Dense, Dropout

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

chars = 'rndeqkstchmavgilfpwybzuxXo'
ctable = CharacterTable(chars, 11)

ACIDS = 26
encoding_dim = 20

np.set_printoptions(threshold=np.nan)

print("Generating data...")

data = []
test = []

dataNames = []

record = SeqIO.parse("bigFile.fa", "fasta")

ind = 0
for rec in record:
    ind +=1
    if len(test) > 229999:
        break
    if ind > 25502:
        break
    if ((len(data) + len(test)) % 6) == 5:
        for k in range(len(rec.seq)//3 - 10):
            test.append([rec.seq[3 * k + i] for i in range(11)])
    else:
        for k in range(len(rec.seq)//3 - 10):
            data.append([rec.seq[3 * k + i] for i in range(11)] )
            dataNames.append(rec.name)     

X = np.zeros((len(data), 11, len(chars)), dtype=np.bool)

for i, sentence in enumerate(data):
    X[i] = ctable.encode(sentence, maxlen=11)

X_val = np.zeros((len(test), 11, len(chars)), dtype=np.bool)

for i, sentence in enumerate(test):
    X_val[i] = ctable.encode(sentence, maxlen=11)

print("Creating model...")
model = Sequential()

#Recurrent encoder
model.add(recurrent.LSTM(encoding_dim, input_shape=(11, ACIDS), dropout_W=0.1, dropout_U=0.1))
#model.add(recurrent.LSTM(encoding_dim, dropout_W=0.1, dropout_U=0.1))

model.add(RepeatVector(11))

#And decoding
model.add(recurrent.LSTM(ACIDS, return_sequences=True))

model.add(TimeDistributed(Dense(ACIDS)))

model.add(Activation('softmax'))

model.load_weights("RecOne.h5")

model.compile(optimizer='rmsprop', loss='binary_crossentropy')

get_summary = K.function([model.layers[0].input], [model.layers[0].output])

print("Let's go!")

print(ind)

Embed = [[0 for _ in range(encoding_dim)] for _ in range(len(X))]

for i in range(len(X)):
    row = X[np.array([i])]
    preds = model.predict_classes(row, verbose=0)
    correct = ctable.decode(row[0])
    intermediate = get_summary([row])[0][0]
    guess = ctable.decode(preds[0], calc_argmax=False)
    Embed[i] = intermediate

Alg = cluster.KMeans(n_clusters=4)

Alg.fit(Embed)
Cluster_ind = Alg.predict(Embed)

Cluster = [[] for _ in range(8)]

for i in range(len(Embed)):
    Cluster[Cluster_ind[i]].append(dataNames[i])

text = open('Names.txt', 'w')

for s in Cluster[0]:
    for c in s:
        text.write(c)
    text.write('\n')
