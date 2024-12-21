from time import sleep, time
from threading import Thread

from .toolbox import ToolBox
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

class GameController:
    is_training = False
    empty_battle_screen = None
    command_queue = []
    
    @staticmethod
    def add_command(command: int, event_reward: dict, event_time: int) -> None:
        command = int(command)
        GameController.command_queue.insert(0, (command, event_reward, event_time))
    
    @staticmethod
    def command_parser(command: int):
        desk_index = (command - 1) // 552
        
        from_x, from_y = desk[desk_index]
        
        command -= 552 * desk_index
        
        if command <= 6:
            to_x, to_y = x_map[6 + command - 1], y_map[0]
        elif command > 546:
            to_x, to_y = x_map[(553 - command)], y_map[-1]
        else:
            to_x, to_y = x_map[(command-7) % 18], y_map[((command-7) // 18) + 1]
        
        return from_x, from_y, to_x, to_y
    
    @staticmethod
    def controller_thread() -> None:
        while True:
            if not GameController.command_queue:
                sleep(0.01)
            else:
                try:
                    command, event_reward, event_time = GameController.command_queue.pop()
                    screen = ScreenCopy.get_image()
                    if not ToolBox.is_in_game(screen):
                        event_reward[event_time] = 0
                        continue
                    if command == 0:
                        if ToolBox.is_full_elixir(screen):
                            event_reward[event_time] = -1
                        else:
                            event_reward[event_time] = 0.01
                        continue
                    from_x, from_y, to_x, to_y = GameController.command_parser(command)
                    AndroidViewClient.click(from_x, from_y)
                    sleep(0.15)
                    
                    elixir_cost = ToolBox.get_elixir_cost(screen, (command - 1) // 552)
                    place_type = ToolBox.get_can_place_type(screen, ScreenCopy.get_image())
                    if to_y < y_map[-6]:
                        event_reward[event_time] = -elixir_cost * 1.2
                    elif to_y < y_map[14]:
                        event_reward[event_time] = elixir_cost * 1.2
                    elif sum(place_type) == 4:
                        event_reward[event_time] = elixir_cost * 1.2
                    elif to_y >= y_map[21]:
                        event_reward[event_time] = -0.2 * elixir_cost
                    elif not place_type[1] and not place_type[3] and to_y > y_map[14]:
                        event_reward[event_time] = -0.2 * elixir_cost
                    elif not place_type[1] and to_x <= x_map[8] and to_y > y_map[14]:
                        event_reward[event_time] = -0.2 * elixir_cost
                    elif not place_type[3] and to_x > x_map[8] and to_y > y_map[14]:
                        event_reward[event_time] = -0.2 * elixir_cost
                    else:
                        event_reward[event_time] = elixir_cost * 1.2
                    
                    AndroidViewClient.click(to_x, to_y)
                    sleep(0.02)
                except Exception as err:
                    raise err
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