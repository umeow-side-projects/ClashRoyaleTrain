import cv2
import logging
import numpy as np

from time import time, sleep

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from .toolbox import ToolBox
from .scrcpy import ScreenCopy
from .game_controller import GameController

class GameEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        # 定義動作空間 (假設最大552種指令對應的動作)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2208, name="action"
        )
        # 定義狀態空間 (用遊戲畫面代表)
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(160, 80), dtype=np.float32, minimum=0.0, maximum=1.0, name="observation"
        )
        self._state = np.zeros((160, 80), dtype=np.float32)  # 初始化狀態為空畫面
        self._episode_ended = False
        
        self._last_game_start_time = 0
        self._game_start = False
        
        self._screen = None
        
        self._right_up_life = 0
        self._right_down_life = 0
        self._left_up_life = 0
        self._left_down_life = 0
        self._up_life = 0
        self._down_life = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """重置環境."""
        self._episode_ended = False
        
        if not self._episode_ended:
            self._last_game_start_time = time()
        
        self._screen = ScreenCopy.get_image()
        self._state = preprocess_image(self._screen)
        self._right_up_life = 0
        self._right_down_life = 0
        self._left_up_life = 0
        self._left_down_life = 0
        self._up_life = 0
        self._down_life = 0
        return ts.restart(self._state)

    def _step(self, action):
        """執行單步動作."""
        if self._episode_ended:
            return self.reset()

        # logging.info('step...')
        while not self._game_start and not ToolBox.is_in_game():
            # logging.info('in step and game not start')
            sleep(0.5)
        
        self._game_start = True

        # 計算回報和是否結束回合
        if ToolBox.is_end(self._screen):
            game_time = time() - self._last_game_start_time
            self._episode_ended = True
            self._game_start = False
            crown_number = ToolBox.count_crown(self._screen)
            reward = crown_number[0] * 200 - crown_number[1] * 200  # 戰鬥結束並勝利，給予回報
            return ts.termination(self._state, time_reward(reward, game_time))

        # 中間回報基於戰鬥情況（可自定）
        reward = self.evaluate_battle_reward(action)
        
        sleep(0.5)
        
        self._screen = ScreenCopy.get_image()
        
        self._state = preprocess_image(self._screen)

        return ts.transition(self._state, reward)
    
    def evaluate_battle_reward(self, action):
        """定義回報規則，例如基於敵人數量或總傷害."""
        # 假設遊戲有方法檢測敵人數或傷害數據
        # reward = ...
        # reward = 0.1  # 測試階段可用固定值代替
        
        desk_index = (action - 1) // 552
        desk_bool = ToolBox.can_place_desk(self._screen)
        
        if not desk_bool[desk_index]:
            return -1
        
        event_time = time()

        GameController.add_command(action, event_reward, event_time)
        
        while not event_time in event_reward:
            sleep(0.01)
        
        reward = event_reward[event_time]
        
        del event_reward[event_time]
        
        # right down
        right_down = self._screen[397, 263:297]
        right_down_life = 0
        for i in range(right_down.shape[1])[::-1]:
            if ToolBox.check_no_life_color(right_down[i], 0):
                right_down_life += 1
            else:
                break
        
        # left down
        left_down = self._screen[397, 75:109]
        left_down_life = 0
        for i in range(left_down.shape[1])[::-1]:
            if ToolBox.check_no_life_color(left_down[i], 0):
                left_down_life += 1
            else:
                break
        
        # left up
        left_up = self._screen[98, 263:297]
        left_up_life = 0
        for i in range(left_up.shape[1])[::-1]:
            if ToolBox.check_no_life_color(left_up[i], 1):
                left_up_life += 1
            else:
                break
        
        # right up
        right_up = self._screen[98, 75:109]
        right_up_life = 0
        for i in range(right_up.shape[1])[::-1]:
            if ToolBox.check_no_life_color(right_up[i], 1):
                right_up_life += 1
            else:
                break
        
        # down
        down = self._screen[483, 162:213]
        down_life = 0
        for i in range(down.shape[1])[::-1]:
            if ToolBox.check_no_life_color(down[i], 2):
                down_life += 1
            else:
                break
        
        # up
        up = self._screen[25, 160:211]
        up_life = 0
        for i in range(up.shape[1])[::-1]:
            if ToolBox.check_no_life_color(up[i], 3):
                up_life += 1
            else:
                break
        
        if left_down_life > self._left_down_life:
            reward -= (left_down_life - self._left_down_life) * 50
            self._left_down_life = left_down_life
        
        if right_down_life > self._right_down_life:
            reward -= (right_down_life - self._right_down_life) * 50
            self._right_down_life = right_down_life
        
        if left_up_life > self._left_up_life:
            reward += (left_up_life - self._left_up_life) * 50
            self._left_up_life = left_up_life
        
        if right_up_life > self._right_up_life:
            reward += (right_up_life - self._right_up_life) * 50
            self._right_up_life = right_up
        
        if down_life > self._down_life:
            reward -= (down_life - self._down_life) * 50
            self._down_life = down_life
        
        if up_life > self._up_life:
            reward += (up_life - self._up_life) * 50
            self._up_life = up_life
        
        return reward

def preprocess_image(image):
    """處理遊戲畫面：灰階化、調整大小、正規化."""
    small_img = cv2.resize(image, (80, 160), interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    return gray_img / 255.0

def time_reward(reward, game_time):
    if game_time < 60:
        return reward * 2
    if game_time < 120:
        return reward * 1.5
    if game_time < 150:
        return reward * 1.2
    if game_time < 160:
        return reward * 1.1
    return reward

event_reward = {}