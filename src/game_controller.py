from typing import Any
import cv2
from nptyping import NDArray, Shape
import numpy as np

from time import sleep, time
from threading import Thread
from .scrcpy import ScreenCopy
from .android_view_client import AndroidViewClient

desk = [
    (340, 1700),
    (540, 1700),
    (740, 1700),
    (940, 1700)]

x_map = [
    107, 158, 208, 258, 309, 361, 412, 464, 516, 567, 617, 668, 719, 769, 820, 870, 922, 974
]

y_map = [
    1455, 1410, 1365, 1325, 1285, 1245, 1200, 1160, 1120, 1080, 1040, 995, 955,
    915, 870, 830, 790, 760, 710, 670, 620, 585, 545, 505, 465, 425, 380, 340,
    300, 265, 230, 185
]

end_screen = cv2.imread('end_screen.png').astype(np.int16)
home_screen = cv2.imread('home_screen.png').astype(np.int16)
combat_screen = cv2.imread('combat_screen.png').astype(np.int16)
combat_menu_screen = cv2.imread('combat_menu_screen.png').astype(np.int16)

class GameController:
    command_queue = []
    
    @staticmethod
    def add_command(command: int, event_reward: dict, event_time: int) -> None:
        GameController.command_queue.insert(0, (command, event_reward, event_time))
    
    @staticmethod
    def can_place(x, y) -> bool:
        if y == 0 or y == 31:
            return 5 < x < 12
        return True
    
    @staticmethod
    def command_parser(command: int):
        from_x, from_y = desk[command // 552]
        
        command %= 552
        if command <= 6:
            to_x, to_y = x_map[6 + command - 1], y_map[0]
        elif command > 546:
            to_x, to_y = x_map[(553 - command)], y_map[-1]
        else:
            to_x, to_y = x_map[(command-7) % 18], y_map[((command-7) // 18) + 1]
        
        return from_x, from_y, to_x, to_y

    @staticmethod
    def is_in_game() -> bool:
        img = ScreenCopy.get_image()
        if abs(img[13][342][0] - 182) > 10:
            return False
        if abs(img[13][342][1] - 231) > 10:
            return False
        if abs(img[13][342][2] - 245) > 10:
            return False
        return True
    
    @staticmethod
    def check_end_color(color_type: int, color: NDArray[Shape['3'], np.uint8]) -> bool:
        if color_type == 0:
            if abs(color[0] - 182) > 20:
                return False
            if abs(color[1] - 100) > 20:
                return False
            if abs(color[2] - 30) > 20:
                return False
        
        if color_type == 1:
            if abs(color[0] - 69) > 20:
                return False
            if abs(color[1] - 31) > 20:
                return False
            if abs(color[2] - 169) > 20:
                return False
            
        return True
    
    @staticmethod
    def is_end() -> bool:
        img = ScreenCopy.get_image()
        target = img[358:378, 157:177, 0:3]
        
        diff = np.abs(end_screen - target)
        
        result = np.all(diff < 20)
        
        return result

    @staticmethod
    def is_in_home_page() -> bool:
        img = ScreenCopy.get_image()
        target = img[602:617, 226:238, 0:3]
        
        diff = np.abs(home_screen - target)
        
        result = np.all(diff < 20)
        
        return result
    
    @staticmethod
    def is_in_combat_page() -> bool:
        img = ScreenCopy.get_image()
        target = img[601:615, 243:255, 0:3]
        
        diff = np.abs(combat_screen - target)
        
        result = np.all(diff < 10)
        
        return result
    
    @staticmethod
    def is_in_combat_menu_page() -> bool:
        img = ScreenCopy.get_image()
        target = img[325:335, 70:80, 0:3]
        
        diff = np.abs(combat_menu_screen - target)
        
        result = np.all(diff < 10)
        
        return result
    
    @staticmethod
    def get_can_place_type(origin_img, img) -> int:
        diff = cv2.absdiff(origin_img, img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        binary_diff: NDArray[Shape[Any, Any], np.uint8] = cv2.threshold(gray_diff, 1, 255, cv2.THRESH_BINARY)[1]

        
        # [176][67] ~ [230][104]
        left_down_block = binary_diff[176:230, 67:104]
        left_down = np.sum(left_down_block == 255) / left_down_block.size > 0.7
        
        # [183][254] ~ [237][291]
        right_down_block = binary_diff[183:237, 254:291]
        right_down = np.sum(right_down_block == 255) / right_down_block.size > 0.7
        
        # [68][63] ~ [84][108]
        left_up_block = binary_diff[68:84, 63:108]
        left_up = np.sum(left_up_block == 255) / left_up_block.size > 0.7
        
        # [68][251] ~ [84][296]
        right_up_block = binary_diff[68:84, 251:296]
        right_up = np.sum(right_up_block == 255) / right_up_block.size > 0.7
        
        # cv2.imwrite('output.png', binary_diff)
        
        return [
            not left_up, not left_down, not right_up, not right_down
        ]
    
    @staticmethod
    def can_place_desk() -> list[bool, bool, bool, bool]:
        img = ScreenCopy.get_image()
        color = np.array([[img[591, 116], img[591, 184], img[591, 252], img[591, 319]]], np.uint8)
        hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0]
        
        can_despatch = [hsv_color[i][1] > 20 and hsv_color[i][0] != 14 for i in range(4)]
        
        # print(can_despatch)
        return can_despatch
    
    @staticmethod
    def check_crown_color(color: NDArray[Shape['3'], np.uint8]):
        if abs(color[0] - 30) > 10:
            return False
        if abs(color[1] - 140) > 10:
            return False
        if abs(color[2] - 240) > 10:
            return False
        return True

    @staticmethod
    def count_crown() -> None | list[int, int]:
        if not GameController.is_end():
            return None
        
        img = ScreenCopy.get_image()
        color1 = img[330, 96]
        color2 = img[322, 179]
        color3 = img[330, 263]
        color4 = img[142, 95]
        color5 = img[134, 178]
        color6 = img[142, 262]
        
        return [
            GameController.check_crown_color(color1) + GameController.check_crown_color(color2) + GameController.check_crown_color(color3),
            GameController.check_crown_color(color4) + GameController.check_crown_color(color5) + GameController.check_crown_color(color6),
        ]
    
    @staticmethod
    def controller_thread() -> None:
        while True:
            if not GameController.command_queue:
                sleep(0.02)
            try:
                command, event_reward, event_time = GameController.command_queue.pop()
                if not GameController.is_in_game():
                    event_reward[event_time] = 0
                    continue
                if command == 0:
                    event_reward[event_time] = 0.02
                    continue
                from_x, from_y, to_x, to_y = GameController.command_parser(command)
                origin_img = ScreenCopy.get_image()
                AndroidViewClient.click(from_x, from_y)
                sleep(0.15)
                
                place_type = GameController.get_can_place_type(origin_img, ScreenCopy.get_image())
                
                if to_y < y_map[14]:
                    event_reward[event_time] = 1
                elif sum(place_type) == 4:
                    event_reward[event_time] = 1
                elif to_y >= y_map[21]:
                    event_reward[event_time] = -0.2
                elif not place_type[1] and not place_type[3] and to_y > y_map[14]:
                    event_reward[event_time] = -0.2
                elif not place_type[1] and to_x <= x_map[8] and to_y > y_map[14]:
                    event_reward[event_time] = -0.2
                elif not place_type[3] and to_x > x_map[8] and to_y > y_map[14]:
                    event_reward[event_time] = -0.2
                else:
                    event_reward[event_time] = 1
                
                AndroidViewClient.click(to_x, to_y)
                sleep(0.02)
            except:
                pass
    
    @staticmethod
    def click_combat_button() -> None:
        AndroidViewClient.click(979, 1838)
    
    @staticmethod
    def click_combat_menu_button() -> None:
        AndroidViewClient.click(182, 1506)
    
    @staticmethod
    def click_combat_start_button() -> None:
        AndroidViewClient.click(538, 1336)
        
    @staticmethod
    def click_exit_combat_button() -> None:
        AndroidViewClient.click(537, 1677)
    
    @staticmethod
    def init() -> None:
        Thread(target=GameController.controller_thread, daemon=True).start()