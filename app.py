import os
from threading import Thread
import cv2
import logging
import inquirer
import coloredlogs
import inquirer.questions

import numpy as np

from src.android_view_client import AndroidViewClient
from src.game_controller import GameController
from src.scrcpy import ScreenCopy
from src.config import Config
from src.train import train_main
from src.driver import auto_game

from time import sleep

logger = logging.getLogger('app.py')

coloredlogs.install(level='DEBUG')

coloredlogs.install(level='DEBUG', logger=logger)

def keep_alive() -> None:
    while True:
        sleep(1000000)

def click_test() -> None:
    while True:
        questions = [
            inquirer.List('mode',
                message="click or swipe?",
                choices=['click', 'swipe'],
            )]
        answer = inquirer.prompt(questions)['mode']
        
        if answer == 'click':
            x, y = map(int, input('input pos(x, y): ').split())
            AndroidViewClient.click(x, y, 3)
        
        if answer == 'swipe':
            from_x, from_y, to_x, to_y = map(int, input('input pos(from_x, from_y. to_x, to_y)').split())
            AndroidViewClient.drag(from_x, from_y, to_x, to_y, 1)

def custom_path() -> None:
    # while True:
    #     print(GameController.is_full_elixir())
    #     sleep(0.5)
    while True:
        command = eval(input('input command: '))
        
        if command == 'screenshot':
            cv2.imwrite('screenshot.png', ScreenCopy.get_image())
            continue
        if command == 'set-homepage':
            cv2.imwrite('home_screen.png', ScreenCopy.get_image()[602:617, 226:238, 0:3])
            continue
        if command == 'set-combatpage':
            cv2.imwrite('combat_screen.png', ScreenCopy.get_image()[601:615, 243:255, 0:3])
            continue
        if command == 'set-combatmenu':
            cv2.imwrite('combat_menu_screen.png', ScreenCopy.get_image()[433:453, 117:137, 0:3])
            continue
        if command == 'set-end':
            cv2.imwrite('end_screen.png', ScreenCopy.get_image()[358:378, 157:177, 0:3])
            continue
        GameController.add_command(command)

def main() -> None:
    execute_path = os.path.abspath('.')
    file_path = os.path.abspath(__file__ + '/..')
    
    if execute_path != file_path:
        logger.error('please run script on project root directory!')
        return
    
    if not AndroidViewClient.check_ready():
        logger.error("please check `CulebraTester2` service exist")
        return
    
    if not Config.check_key('arch_name'):
        questions = [
            inquirer.List('arch_name',
                message="PC architecture? (scrcpy only support these architecture)",
                choices=['linux-x86_64', 'macos-aarch64', 'macos-x86_64', 'win32', 'win64'],
            )]
        
        answer = inquirer.prompt(questions)['arch_name']
        Config.set('arch_name', answer)
    
    if not ScreenCopy.check_support():
        logger.error("the `arch_name` key in `Config` is not support architecture!")
        return
    
    logger.info('Initializing ViewClient...')
    AndroidViewClient.init()
    
    logger.info('Initializing ScrCpy...')
    ScreenCopy.init()
    
    logger.info('Initializing GameController...')
    GameController.init()
    
    logger.info('Initializing Done!')
    
    # while not GameController.is_in_home_page():
    #     sleep(0.1)
        
    # GameController.click_combat_button()
    
    # while not GameController.is_in_combat_page():
    #     sleep(0.1)
    
    # GameController.click_combat_menu_button()
    
    # while not GameController.is_in_combat_menu_page():
    #     sleep(0.1)
    
    # GameController.click_combat_start_button()
    
    # while not GameController.is_end():
    #     sleep(0.1)
    
    # print(GameController.count_crown())
    # GameController.click_exit_combat_button()
    
    # while not GameController.is_in_combat_page():
    #     sleep(1)
    
    # print('done!')
    
    # while True:
    #     print(GameController.count_crown())
    #     sleep(0.1)
    # cv2.imwrite('test.png', ScreenCopy.get_image())
    
    # custom_path()
    # click_test()
    
    Thread(target=auto_game, daemon=True).start()
    train_main()



if __name__ == '__main__':
    main()