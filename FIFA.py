import numpy as np
import pytesseract as pt
import cv2
from CNN import CNN
from PIL import Image
from grabscreen import grab_screen
from directkeys import *
import sys

class FIFA(object):
    """
    This class acts as the intermediate "API" to the actual game. Double quotes API because we are not touching the
    game's actual code. It interacts with the game simply using screen-grab (input) and keypress simulation (output)
    using some clever python libraries.
    """

    cnn_graph = CNN()
    reward = 0

    def __init__(self):
        self.reset()

    def _get_reward(self, action):
        screen = self.capture_screen()
        screen_resized = cv2.resize(screen, (780, 480))

        # the reward meter at top right corner of game screen
        reward_screen = screen[95:125, 1135:1190]
        reward_screen = np.pad(reward_screen, ((100, 100), (100, 100), (0, 0)), 'edge')
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')
        try:
            ocr_result = pt.image_to_string(i)
            ingame_reward = int(''.join(c for c in ocr_result if c.isdigit()))

            print('current reward: ' + str(self.reward))
            print('observed reward: ' + str(ingame_reward))
            if ingame_reward - self.reward >= 1000:
                # if ball hits the target
                self.reward = ingame_reward
                ingame_reward = 1
            elif ingame_reward - self.reward > 0:
                # if ball hits pole
                self.reward = ingaume_reward
                ingame_reward = 0
            elif ingame_reward - self.reward == 0:
                # if missed
                self.reward = ingame_reward
                ingame_reward = -1
            else:
                #no idea whats happning here
                ingame_reward = 0
            print('q-learning reward: ' + str(ingame_reward))
        except:
            ingame_reward = -1 if self._is_over(action) else 0
            print("Unexpected error:", sys.exc_info()[0])
            print('exception q-learning reward: ' + str(ingame_reward))

        return ingame_reward

    def _is_over(self, action):
        # Check if the ball is still there to be hit. If ball is still present in the screenshot, game isn't over yet.
        # What follows is arguably the most sophisticated way to find that out.
        # screen = grab_screen(region=None)
        # screen = screen[25:-40, 1921:]
        # ball_location = screen[790:830, 940:980]
        # # Check red channel (rgb) for grass or ball using threshold 60
        # is_over = np.mean(ball_location[:, :, 0]) < 60
        # print('is over, ball presence. mean=' + str(np.mean(ball_location[:, :, 0])))
        is_over = True if action in [0, 1] else False
        return is_over

    def capture_screen(self):
        screen = grab_screen(region=None)
        return screen[: 750, : 1290]

    def observe(self):
        print('\nobserve')
        # get current state s from screen using screen-grab
        screen = self.capture_screen()

        # if drill over, restart drill and take screenshot again
        restart_button = screen[559:581, 390:481]
        restart_button = np.pad(restart_button, ((50, 50), (50, 50), (0, 0)), 'edge')
        i = Image.fromarray(restart_button.astype('uint8'), 'RGB')
        restart_text = pt.image_to_string(i)
        if "RETRY DRILL" in restart_text:
            # press enter key
            print('Pressing enter, reset reward')
            self.reward = 0
            # PressKey(leftarrow)
            # time.sleep(0.4)
            # ReleaseKey(leftarrow)
            PressKey(enter)
            time.sleep(0.1)
            ReleaseKey(enter)
            time.sleep(1)
            screen = self.capture_screen()

        # process through CNN to get the feature map from the raw image
        state = self.cnn_graph.get_image_feature_map(screen)
        return state

    def act(self, action):
        display_action = ['shoot_low', 'shoot_high', 'left_arrow', 'right_arrow']
        print('action: ' + str(display_action[action[0]]) + ' + ' + str(display_action[action[1]]))
        # [ shoot_low, shoot_high, left_arrow, right_arrow ]
        keys_to_press = np.array([[spacebar], [spacebar], [U], [S]])
        # need to keep all keys pressed for some time before releasing them otherwise fifa considers them as accidental
        # key presses.
        for key in keys_to_press[action]:
            PressKey(key)

        time.sleep(0.2) if 1 in action else time.sleep(0.05)

        for key in keys_to_press[action]:
            ReleaseKey(key)

        # wait until some time after taking action
        time.sleep(1)

        reward = self._get_reward(action)
        game_over = self._is_over(action)
        return self.observe(), reward, game_over

    def reset(self):
        return
