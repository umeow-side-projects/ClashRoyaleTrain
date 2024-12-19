from time import sleep
from .game_controller import GameController

def auto_game() -> None:
    first = True
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
            
            if not first:
                while not GameController.is_training:
                    sleep(0.1)
                
                while GameController.is_training:
                    sleep(0.1)
            
            click_timer = 0
            while not GameController.is_in_game():
                if click_timer == 0:
                    GameController.click_combat_start_button()
                    click_timer= 100
                else:
                    click_timer -= 1
                sleep(0.1)
            
            first = False
            
            while not GameController.is_end():
                sleep(0.5)
                
            sleep(2)
            GameController.click_exit_combat_button()
        except:
            pass