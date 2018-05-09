import numpy as np
import pytesseract as pt
import cv2
from PIL import Image
from grabscreen import grab_screen
from directkeys import *


class FIFA(object):
    """
    Class FIFA is the intermediate "API" to the actual game.
    It interacts with the actual game using screen-grab (input) and keypress simulation (output) using python libraries.
    """

    def __init__(self):
        self.reset()

    def _get_reward(self):
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        screen_resized = cv2.resize(screen, (780, 480))

        # the reward meter at top right corner of game screen
        reward_screen = screen[85:130, 1650:1730]
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        total_reward = pt.image_to_string(i)
        return total_reward

    def _is_over(self):
        return True

    def observe(self):
        # get current state s from screen using screen-grab
        screen = grab_screen(region=None)
        screen = screen[25:-40, 1921:]
        # process through CNN - assume for now screen_resized is the 128d output
        screen_resized = cv2.resize(screen, (780, 480))
        return screen_resized.reshape((1, -1))

    def act(self, action):
        PressKey(0x11)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        self.reward = 0