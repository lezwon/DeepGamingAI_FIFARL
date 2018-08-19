import time

import numpy as np

from getkeys import key_check

# parameters
num_actions = 4  # [ shoot_low, shoot_high, left_arrow, right_arrow]
max_memory = 500  # Maximum number of experiences we are storing
hidden_size = 100  # Size of the hidden layers
batch_size = 1  # Number of experiences we use for training per batch
grid_size = 10  # Size of the playing field
timesteps = 20

def predict(model, state_t):
    input = np.zeros((1, timesteps, ) + state_t.shape)
    input[0][timesteps - 1] = state_t
    q = model.predict(input)
    # We pick the action with the highest expected reward
    return q[0][-1].argsort()[-2:][::-1]

def test(game, model, epochs, verbose=1):
    # Train
    # Reseting the win counter
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []

    # logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

    # Epochs is the number of games we play
    for e in range(epochs):

        game.reset()
        game_over = False
        input_t = game.observe()
        if e == 0:
            paused = True
            print(
                'Training is paused. Press p once game is loaded and is ready to be played.')
        else:
            paused = False

        if not paused:
            time.sleep(1)
            frame_count = 0
            game.start_drill()

            while not game.drill_over():

                input_tm1 = input_t
                action = predict(model, input_tm1)
                input_t = game.act(action)
                frame_count += 1
                print(u'Frame Count: ' + str(frame_count))

            reward = game._get_reward()
            if reward == 1:
                win_cnt += 1

        # menu control
        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
        elif 'O' in keys:
            print('Quitting!')
            return

    return win_hist
