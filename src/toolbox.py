import cv2
import pickle
import numpy as np

from nptyping import NDArray, Shape
from typing import Any

from .scrcpy import ScreenCopy
from .image_database import ImageDatabase

model_file = open('model.pickle', 'rb')
model = pickle.load(model_file)

full_elixir_screen = ImageDatabase.get_image('full_elixir_screen.png').astype(np.int16)
combat_menu_screen = ImageDatabase.get_image('combat_menu_screen.png').astype(np.int16)
combat_screen = ImageDatabase.get_image('combat_screen.png').astype(np.int16)
home_screen = ImageDatabase.get_image('home_screen.png').astype(np.int16)
end_screen = ImageDatabase.get_image('end_screen.png').astype(np.int16)

class ToolBox:
    @staticmethod
    def get_elixir_cost(screen: cv2.Mat, desk_index: int):
        img = screen.copy()
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if desk_index == 0:
            img = img[594:607, 108:118]
        elif desk_index == 1:
            img = img[594:607, 174:184]
        elif desk_index == 2:
            img = img[594:607, 242:252]
        elif desk_index == 3:
            img = img[594:607, 310:320]
        
        x = np.array([img.ravel() / 255.0]).T
        return model.predict(x)[0] + 2
    
    @staticmethod
    def is_in_combat_menu_page(screen: cv2.Mat | None = None) -> bool:
        if screen is None:
            screen = ScreenCopy.get_image()
            
        target = screen[433:453, 117:137, :]
        
        diff = np.abs(combat_menu_screen - target)
        # print(diff)
        
        result = np.all(diff < 20)
        
        return result
    
    @staticmethod
    def is_in_combat_page(screen: cv2.Mat | None = None) -> bool:
        if screen is None:
            screen = ScreenCopy.get_image()
            
        target = screen[601:615, 243:255, :]
        
        diff = np.abs(combat_screen - target)
        
        result = np.all(diff < 20)
        
        return result

    @staticmethod
    def is_in_home_page(screen: cv2.Mat | None = None) -> bool:
        if screen is None:
            screen = ScreenCopy.get_image()
            
        target = screen[602:617, 226:238, :]
        
        diff = np.abs(home_screen - target)
        
        result = np.all(diff < 20)
        
        return result
    
    @staticmethod
    def is_end(screen: cv2.Mat | None = None) -> bool:
        if screen is None:
            screen = ScreenCopy.get_image()
            
        target = screen[358:378, 157:177, :]
        
        diff = np.abs(end_screen - target)
        
        result = np.all(diff < 20)
        
        return result
    
    @staticmethod
    def is_full_elixir(screen: cv2.Mat) -> bool:
        target = screen[612:615, 95:125, :]
        
        diff = np.abs(full_elixir_screen - target)
        
        result = np.all(diff < 20)
        
        return result
    
    @staticmethod
    def can_place_desk(screen: cv2.Mat) -> list[bool, bool, bool, bool]:
        color = np.array([[screen[591, 116], screen[591, 184], screen[591, 252], screen[591, 319]]], np.uint8)
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
    def count_crown(screen: cv2.Mat) -> None | list[int, int]:
        color1 = screen[330, 96]
        color2 = screen[322, 179]
        color3 = screen[330, 263]
        color4 = screen[142, 95]
        color5 = screen[134, 178]
        color6 = screen[142, 262]
        
        return [
            ToolBox.check_crown_color(color1) + ToolBox.check_crown_color(color2) + ToolBox.check_crown_color(color3),
            ToolBox.check_crown_color(color4) + ToolBox.check_crown_color(color5) + ToolBox.check_crown_color(color6),
        ]
    
    @staticmethod
    def is_in_game(screen: cv2.Mat | None = None) -> bool:
        if screen is None:
            screen = ScreenCopy.get_image()
        
        img = screen.astype(np.int16)
            
        if abs(img[13][342][0] - 182) > 10:
            return False
        if abs(img[13][342][1] - 231) > 10:
            return False
        if abs(img[13][342][2] - 245) > 10:
            return False
        return True
    
    @staticmethod
    def get_can_place_type(origin_img: cv2.Mat, img: cv2.Mat) -> int:
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
    def check_no_life_color(color, mode: int) -> bool:
        if mode == 0:
            if abs(color[0] - 117) > 20:
                return False
            if abs(color[1] - 80) > 20:
                return False
            if abs(color[2] - 58) > 20:
                return False
        elif mode == 1:
            if abs(color[0] - 72) > 20:
                return False
            if abs(color[1] - 56) > 20:
                return False
            if abs(color[2] - 94) > 20:
                return False
        elif mode == 2:
            if abs(color[0] - 86) > 20:
                return False
            if abs(color[1] - 96) > 20:
                return False
            if abs(color[2] - 105) > 20:
                return False
        elif mode == 3:
            if abs(color[0] - 35) > 20:
                return False
            if abs(color[1] - 49) > 20:
                return False
            if abs(color[2] - 71) > 20:
                return False
        return True
    