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
        self.max_score = 0
        self.score = 0

    def save_image(self, numpy_image = None):

        if numpy_image is None:
            numpy_image = self.capture_screen()

        i = Image.fromarray(numpy_image.astype('uint8'), 'RGB')
        try:
            i.save('image.jpg')
            return pt.image_to_string(i)
        except:
            print("Unexpected error:", sys.exc_info()[0])


    def record_score(self):
        screen = self.capture_screen()

        # the reward meter at top right corner of game screen
        reward_screen = screen[87:125, 1120:1200]
        reward_screen = np.where(reward_screen >= 110, 0, 255)
        # reward_screen = np.pad( reward_screen, ((100, 100), (100, 100), (0, 0)), 'edge')
        i = Image.fromarray(reward_screen.astype('uint8'), 'RGB')

        try:
            ocr_result = pt.image_to_string(i)
            current_score = int(''.join(c for c in ocr_result if c.isdigit()))
            self.score = current_score
        except:
            current_score = 0
            print("Unexpected error:", sys.exc_info()[0])
        finally:
            print('exception q-learning reward: ' + str(current_score))
            print(f'Current Score is {current_score}')

    def _get_reward(self):
        # screen = self.capture_screen()
        # performance = screen[232:266, 625:765]
        # performance = np.pad(
        #     performance, ((100, 100), (100, 100), (0, 0)), 'edge')
        # i = Image.fromarray(performance.astype('uint8'), 'RGB')

        # try:
        #     ocr_result = pt.image_to_string(i)
        #     if ocr_result == "TRY AGAIN!":
        #         reward = -1:
        #     elif ocr_result == "TRY HARDER!"
        #         reward = 0
        #     else:
        #         reward = 1
        # except:
        #     reward = 0
        #     print("Unexpected error:", sys.exc_info()[0])
        # finally:
        #     print(f'Reward: {reward}')

        if self.score >= 4000 or self.score > self.max_score:
            current_reward = 1
        elif self.score >= 3500:
            current_reward = 0
        else:
            current_reward = -1
        self.max_score = self.score if self.score > self.max_score else self.max_score
        print('q-learning reward: ' + str(current_reward))
        return current_reward

    def _is_over(self, action):
        is_over = True if action in [0, 1] else False
        return is_over

    def drill_over(self):
        screen = self.capture_screen()

        # if drill over, restart drill and take screenshot again
        restart_button = screen[553:580, 380:500]
        restart_button = np.pad(
            restart_button, ((50, 50), (50, 50), (0, 0)), 'edge')
        i = Image.fromarray(restart_button.astype('uint8'), 'RGB')
        restart_text = pt.image_to_string(i)
        return True if "RETRY DRILL" == restart_text else False

    def start_drill(self):
        # press enter key
        print('Pressing enter, reset reward')
        PressKey(enter)
        time.sleep(0.1)
        ReleaseKey(enter)
        time.sleep(1)

    def capture_screen(self):
        screen = grab_screen(region=None)
        return screen[: 750, : 1290]

    def observe(self):
        # process through CNN to get the feature map from the raw image
        screen = self.capture_screen()
        state = cv2.resize(screen, (256, 256))
        # state = self.cnn_graph.get_image_feature_map(resized_screen)
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

        reward = self.record_score()
        return self.observe()

    def reset(self):
        return
