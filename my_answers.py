import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import re

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    index = 0
    step_size = 1


    while (index < (len(series) - window_size )) :
        X.append(series[index:(index+window_size)])
        y.append(series[index+window_size])
        index += step_size
    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y


def test_window_transform_series():
    #for easier debugging
    dataset = np.loadtxt('datasets/normalized_apple_prices.csv')
    window_size = 7
    X, y = window_transform_series(series=dataset, window_size=window_size)

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    #Use regex to remove punctuation
    regex = re.compile('[^a-zA-Z!,.:;? ]')
    return regex.sub('', text)

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    index = 0
    while(index < (len(text) - window_size)):
        inputs.append(text[index:index+window_size])
        outputs.append(text[index + window_size])
        index += step_size

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=((window_size,num_chars) )))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
