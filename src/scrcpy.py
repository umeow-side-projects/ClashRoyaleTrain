import os
import mss
import numpy as np
import atexit
import threading
import pywinctl as pwc

from .config import Config

from time import sleep
from typing import Any
from subprocess import Popen, PIPE
from nptyping import NDArray, Shape

class ScreenCopy:
    ready: bool = False
    arch_name: str = Config.get('arch_name')
    process : None | Popen = None
    window = None
    origin_window_size : None | tuple[int, int]= None
    
    _latest_img = None
    
    @staticmethod
    def at_exit() -> None:
        if ScreenCopy.process:
            ScreenCopy.process.kill()
    
    @staticmethod
    def check_support() -> bool:
        ScreenCopy.arch_name = Config.get('arch_name')
        try:
            for dir_name in os.listdir('scrcpy'):
                if dir_name.startswith(f'scrcpy-{ScreenCopy.arch_name}'):
                    return True
        except Exception:
            pass
        return False
    
    @staticmethod
    def get_executable_file_path() -> str:
        for dir_name in os.listdir('scrcpy'):
            if dir_name.startswith(f'scrcpy-{ScreenCopy.arch_name}'):
                if os.path.exists(f'scrcpy/{dir_name}/scrcpy.exe'):
                    return os.path.abspath(f'scrcpy/{dir_name}/scrcpy.exe')
                return os.path.abspath(f'scrcpy/{dir_name}/scrcpy')
            
    @staticmethod
    def window_protect() -> None:
        while True:
            if not pwc.getWindowsWithTitle('ScreenCopy'):
                os._exit(1)
                
            if ScreenCopy.window.isMinimized:
                ScreenCopy.window.restore()
            sleep(0.01)
    
    @staticmethod
    def image_updater() -> None:
        with mss.mss() as sct:
            while not ScreenCopy.window:
                sleep(0.01)
            
            bbox = (
                ScreenCopy.window.left,
                ScreenCopy.window.top,
                ScreenCopy.window.right,
                ScreenCopy.window.bottom
            )
            ScreenCopy._latest_img = sct.grab(bbox)
            
            ScreenCopy.ready = True
            
            while True:
                try:
                    if not ScreenCopy.window:
                        continue
                    
                    bbox = (
                        ScreenCopy.window.left,
                        ScreenCopy.window.top,
                        ScreenCopy.window.right,
                        ScreenCopy.window.bottom
                    )
                    
                    ScreenCopy._latest_img = sct.grab(bbox)
                    sleep(1 / 120)
                except:
                    pass
    
    @staticmethod
    def init() -> Any:
        ScreenCopy.process = Popen([ScreenCopy.get_executable_file_path(), '-b200M', '-m640', '--window-title=ScreenCopy', '--always-on-top', '--window-borderless', '--no-audio', '--no-control'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        while not pwc.getWindowsWithTitle('ScreenCopy'):
            sleep(0.1)
        sleep(0.1)
        ScreenCopy.window = pwc.getWindowsWithTitle('ScreenCopy')[0]
        ScreenCopy.origin_window_size = ScreenCopy.window.size
        atexit.register(ScreenCopy.at_exit)
        threading.Thread(target=ScreenCopy.window_protect, daemon=True).start()
        threading.Thread(target=ScreenCopy.image_updater, daemon=True).start()
        
        while not ScreenCopy.ready:
            sleep(0.01)
        
    @staticmethod
    def get_image() -> NDArray[Shape["640, Any, 4"], np.uint8]:
        img = np.array(ScreenCopy._latest_img, dtype=np.uint8)
        return img[:, :, :3]