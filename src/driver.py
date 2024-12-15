from time import sleep
from .game_controller import GameController


def auto_game() -> None:
    while not GameController.is_in_home_page():
        sleep(0.1)
    GameController.click_combat_button()
    while True:
        try:
            while not GameController.is_in_combat_page():
                sleep(0.1)
            GameController.click_combat_menu_button()
            while not GameController.is_in_combat_menu_page():
                sleep(0.1)
            GameController.click_combat_start_button()
            while not GameController.is_end():
                sleep(0.5)
            sleep(2)
            GameController.click_exit_combat_button()
        except:
            pass