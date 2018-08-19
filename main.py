import numpy as np
import pytesseract as pt
from keras.layers.core import Dense
from keras.layers import *
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import adam
from matplotlib import pyplot as plt
from FIFA import FIFA
from train import train
from test import test

timesteps = 20
num_actions = 4
grid_sizs  = 256

def baseline_model(grid_size, num_actions):
    model = Sequential()
    model.add(InputLayer( batch_input_shape=(None, timesteps, grid_size, grid_size, 3)))
    model.add(ConvLSTM2D(8, (3, 3), return_sequences = True))
    model.add(MaxPooling3D((1, 2, 2)))
    model.add(ConvLSTM2D(8, (3, 3), return_sequences=True))
    model.add(MaxPooling3D((1, 2, 2)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(num_actions, activation='relu')))
    model.compile(adam(lr=.01), "categorical_crossentropy")
    return model


def moving_average_diff(a, n=100):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_model():
    # load json and create model
    json_file = open('model_epoch1000/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_epoch1000/model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='mse', optimizer='adam')
    return loaded_model


model = baseline_model(grid_size=256, num_actions=4)
# model = load_model()
# model.summary()

# necessary evil

pt.pytesseract.tesseract_cmd = 'tesseract'
# pt.pytesseract.tesseract_cmd = 'I:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

game = FIFA()
print("game object created")

epoch = 1000  # Number of games played in training, I found the model needs about 4,000 games till it plays well

train_mode = 1

if train_mode == 1:
    # Train the model
    hist = train(game, model, epoch, verbose=1)
    print("Training done")
else:
    # Test the model
    hist = test(game, model, epoch, verbose=1)

print(hist)
np.savetxt('win_history.txt', hist)
plt.plot(moving_average_diff(hist))
plt.ylabel('Average of victories per game')
plt.show()
