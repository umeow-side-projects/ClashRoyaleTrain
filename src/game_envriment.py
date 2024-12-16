import cv2
import numpy as np

from time import time, sleep

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

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

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """重置環境."""
        while not GameController.is_in_game():
            sleep(0.5)
        
        self._episode_ended = not GameController.is_in_game()
        self._state = preprocess_image(ScreenCopy.get_image())
        return ts.restart(self._state)

    def _step(self, action):
        """執行單步動作."""
        
        if self._episode_ended:
            return self.reset()

        # 更新環境狀態
        self._state = preprocess_image(ScreenCopy.get_image())

        # 計算回報和是否結束回合
        if GameController.is_end():
            self._episode_ended = True
            crown_number = GameController.count_crown()
            reward = crown_number[0] * 2000 - crown_number[1] * 2000  # 戰鬥結束並勝利，給予回報
            return ts.termination(self._state, reward)

        # 中間回報基於戰鬥情況（可自定）
        reward = evaluate_battle_reward(action)

        return ts.transition(self._state, reward)

def preprocess_image(image):
    """處理遊戲畫面：灰階化、調整大小、正規化."""
    small_img = cv2.resize(image, (80, 160), interpolation=cv2.INTER_AREA)
    
    gray_image = cv2.cvtColor(small_img, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    
    return gray_image / 255.0

event_reward = {}

def evaluate_battle_reward(action):
    """定義回報規則，例如基於敵人數量或總傷害."""
    # 假設遊戲有方法檢測敵人數或傷害數據
    # reward = ...
    # reward = 0.1  # 測試階段可用固定值代替
    
    desk_index = (action - 1) // 552
    desk_bool = GameController.can_place_desk()
    
    if not desk_bool[desk_index]:
        return -1
    
    event_time = time()
    
    GameController.add_command(action, event_reward, event_time)
    
    while not event_time in event_reward:
        sleep(0.01)
    
    reward = event_reward[event_time]
    
    del event_reward[event_time]
    
    sleep(0.5)
    return reward