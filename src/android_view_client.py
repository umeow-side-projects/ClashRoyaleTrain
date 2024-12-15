import logging
import requests

from time import sleep
from typing import Any
from com.dtmilano.android.viewclient import ViewClient
from com.dtmilano.android.uiautomator.uiautomatorhelper import UiAutomatorHelper

logger = logging.getLogger(__file__)

class AndroidViewClient:
    helper : None | UiAutomatorHelper = None
    
    @staticmethod
    def check_ready() -> bool:
        try:
            res = requests.get("http://localhost:9987/", timeout=5)
        
            return res.text.startswith("CulebraTester2")
        except requests.RequestException:
            return False
    
    @staticmethod
    def init() -> Any:
        AndroidViewClient.helper = ViewClient.view_client_helper()
        AndroidViewClient.helper.ui_device.click(0, 0)
        AndroidViewClient.helper.target_context.start_activity('com.supercell.clashroyale', 'com.supercell.titan.GameApp')
        sleep(0.5)
    
    @staticmethod
    def _check_helper(func):
        def wrapper(*args, **kwargs):
            if not AndroidViewClient.helper:
                return
            try:
                func(*args, **kwargs)
            except:
                logger.error('Couldn\'t INJECT EVENT, has something touching screen...')
        return wrapper
    
    @staticmethod
    @_check_helper
    def click(x: int, y: int, duration: float = 0) -> None:
        if duration == 0:
            AndroidViewClient.helper.ui_device.click(x, y)
            return
        
        AndroidViewClient.helper.ui_device.swipe(segments=[(x, y), (x, y)], segment_steps=int(duration * 40))
    
    @staticmethod
    @_check_helper
    def drag(from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 0) -> None:
        AndroidViewClient.helper.ui_device.swipe(segments=[(from_x, from_y), (to_x, to_y)], segment_steps=int(duration * 40))
        
    @staticmethod
    @_check_helper
    def test_drag(from_x: int, from_y: int, to_x: int, to_y: int, duration: float = 1) -> None:
        AndroidViewClient.helper.ui_device.swipe(segments=[(from_x, from_y), (from_x, from_y), (to_x, to_y)], segment_steps=int(duration * 40))
    
    @staticmethod
    @_check_helper
    def path(pathes: tuple[tuple[int, int]], duration: float) -> None:
        AndroidViewClient.helper.ui_device.swipe(segments=pathes, segment_steps=int(duration * 40))