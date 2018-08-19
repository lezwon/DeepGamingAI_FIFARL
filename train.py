import numpy as np
import time
import random
from getkeys import key_check
from ExperienceReplay import ExperienceReplay
import logging

LOG_FILENAME = 'model_epoch1000/train.log'

# parameters
# epsilon = .2  # exploration
num_actions = 4  # [ shoot_low, shoot_high, left_arrow, right_arrow]
max_memory = 1000  # Maximum number of experiences we are storing
batch_size = 4  # Number of experiences we use for training per batch
timesteps = 20
exp_replay = ExperienceReplay(max_memory=max_memory)
history_store = []

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_epoch1000/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_epoch1000/model.h5")
    # print("Saved model to disk")

def predict(model, state_t):
    input = np.zeros((1, timesteps, ) + state_t.shape)
    input[0][timesteps - 1] = state_t
    q = model.predict(input)
    # We pick the action with the highest expected reward
    return q[0][-1].argsort()[-2:][::-1]


def train(game, model, epochs, verbose=1):
    frame_count = 0
    # Train
    # Reseting the win counter
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []

    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

    # Epochs is the number of games we play
    for e in range(epochs):
        loss = 0.
        epsilon = 4 / ((e + 1) ** (1 / 2))
        # Resetting the game
        game.reset()
        game_over = False
        # get tensorflow running first to acquire cudnn handle
        input_t = game.observe()
        if e == 0:
            paused = True
            print('Training is paused. Press p once game is loaded and is ready to be played.')
        else:
            paused = False

        if not paused:
            time.sleep(1)
            sequence = []
            frame_count = 0
            game.start_drill()

            while not game.drill_over():

                

                # The learner is acting on the last observed game screen
                # input_t is a vector containing representing the game screen
                input_tm1 = input_t

                """
                We want to avoid that the learner settles on a local minimum.
                Imagine you are eating in an exotic restaurant. After some experimentation you find 
                that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
                food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
                It's simple: Sometimes, they just don't let you choose but order something random from the menu.
                Maybe you'll like it.
                The chance that your friends order for you is epsilon
                """
                if np.random.rand() <= epsilon:
                    # Eat something random from the menu
                    action = random.sample(range(num_actions), 2)
                    print('random action')
                else:
                    # Choose yourself
                    # q contains the expected rewards for the actions
                    action = predict(model, input_tm1)
                    # We pick the action with the highest expected reward

                # action = predict(model, input_tm1)

                # apply action, get rewards and new state
                input_t = game.act(action)
                # If we managed to catch the fruit we add 1 to our win counter
                sequence.append((input_t, action))
                frame_count += 1
                print(u'Frame Count: ' + str(frame_count))

            reward = game._get_reward()
            if reward == 1:
                win_cnt += 1

            """
            The experiences < s, a, r, s’ > we make during gameplay are our training data.
            Here we first save the last experience, and then load a batch of experiences to train our model
            """

            # store experience
            if len(sequence) >= timesteps:
                np_sequence = np.array(sequence[:timesteps])
            else:
                sequence_length = len(sequence)
                extra = timesteps - sequence_length
                dummy_image = np.zeros_like(input_t)
                dummy_action= [0,0]
                dummy_data = np.array([(dummy_image, dummy_action)] * extra)
                np_sequence = np.array(sequence)
                np_sequence = np.concatenate((dummy_data, np_sequence))
                    
            exp_replay.remember(np_sequence, reward, game_over)

            # Load batch of experiences
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            # train model on experiences
            history = model.fit(inputs, targets)

            # print(loss)
            history_store.append(history)
            loss = sum(history.history['loss'])

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

        if verbose > 0:
            log_message = "Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {} | Epsilon {}".format(e, epochs, loss, win_cnt, epsilon)
            print(log_message)
            logging.info(log_message)
        save_model(model)
        win_hist.append(win_cnt)
    return win_hist
