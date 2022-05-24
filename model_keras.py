# Import dependencies
import pandas as pd
import numpy as np
import sklearn.externals as extjoblib
from sklearn import preprocessing
from sklearn import utils
import joblib


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset in a dataframe object and include only four features as mentioned
url = "http://localhost:5000/api/train-data"
df = pd.read_csv(url, header=None)

dataset = loadtxt(df)
data = df.astype('float32')
print(data)
# split into input (X) and output (y) variables
X = dataset[:,0:436]
y = dataset[:,436]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=446, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=50, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))