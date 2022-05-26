# Import dependencies
import pandas as pd
import numpy as np
import sklearn.externals as extjoblib
from sklearn.preprocessing import MinMaxScaler
from sklearn import utils
import joblib


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Normalization
from keras.optimizers import Adam

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

# Load the dataset in a dataframe object and include only four features as mentioned
url = "http://localhost:5000/api/train-data"
file = '1653549851017.csv'
df = pd.read_csv(file, low_memory=False)

# dataset = loadtxt(df)
data = df.astype('float32')
columnNames = df.columns.tolist()
data = normalize(data)
print(data)

# split into input (X) and output (y) variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
cols=X.shape[1]
print(y)
# define the keras model
model = Sequential()
model.add(Dense(cols, input_shape=(cols,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
opt = Adam(learning_rate=0.01)
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=1, validation_split=0.1)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))