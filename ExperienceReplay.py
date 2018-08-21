import numpy as np
import os

timesteps = 30

class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory.
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """

    def __init__(self, max_memory=1000, discount=.9):
        """
        Setup
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience

        In the memory the information whether the game ended at the state is stored seperately in a nested array
        [...
        [experience, game_over]
        [experience, game_over]
        ...]
        """
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, reward, game_over):
        # Save a state to memory
        self.memory.append(np.array([states, reward, game_over]))
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=3):

        self.model = model

        # How many experiences do we have?
        len_memory = len(self.memory)

        batch_size = len_memory if len_memory < batch_size else batch_size

        # Calculate the number of actions that can possibly be taken in the game
        num_actions = model.output_shape[-1]

        # Dimensions of the game field
        # [states, reward, game_over] -> [input, action] -> input
        env_dim = self.memory[0][0][0][0].shape

        # We want to return an input and target vector with inputs from an observed state...
        inputs = np.empty((batch_size, timesteps ) + env_dim)

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = []

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory, size=batch_size)):

            sequence, reward_t, game_over = self.memory[idx]

            frames = sequence[:, 0]
            actions = np.array(sequence)[:, 1]

            if reward_t == 1:
                # make all actions = 1
                result = np.zeros((timesteps, num_actions))
                change_to = 1
            else:
                # make all actions 0
                result = np.ones((timesteps, num_actions))
                change_to = 0

            for idx, action in enumerate(actions):
                result[idx][action] = change_to

            inputs[i] = np.array([frame for idx, frame in enumerate(frames)])
            targets.append(result)
        return np.array(inputs), np.array(targets)
