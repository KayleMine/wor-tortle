import sys
import os
import time
import ctypes
import requests
from os import system, getpid
from sys import exit
from multiprocessing import Process, Manager, Event, Lock, freeze_support

from functions.processes import get_process, wait_for_process

from settings import Settings


def main_func(scraper_process, ballistics_process, aim_process, render_process, window_process, embed_path, loader_name):
    try:
        print("[>] Welcome to WC Loader")
        if Settings.Memory.is_debug is True:
            print("[>] Current memory method: " + str(Settings.Memory.memory_method))

        if Settings.MultiProcessing.is_debug:
            print("[>] Main process pid: " + str(getpid()))
        manager = Manager()
        shared_lock = Lock()
        shared_exit = Event()

        scraper_shared = manager.dict()
        scraper_shared['value'] = {'camera_ptr': None, 'local_unit': None, 'enemy_units': {}, 'entity_list_count': 0, 'all_units_ptrs':[], 'perform_units': [], 'skip_unit_ptrs': []}

        ballistics_shared = manager.dict()
        ballistics_shared['value'] = {}

        window_shared = manager.dict()
        window_shared['value'] = {}

        window_process_args = (window_shared, shared_lock, shared_exit)
        window = Process(target=window_process, args=window_process_args)
        window.start()

        scraper_process_args = (scraper_shared, window_shared, shared_lock, shared_exit)
        scraper = Process(target=scraper_process, args=scraper_process_args)
        scraper.start()

        ballistics_process_args = (scraper_shared, ballistics_shared, window_shared, shared_lock, shared_exit)
        ballistics = Process(target=ballistics_process, args=ballistics_process_args)
        ballistics.start()

        aim_process_args = (scraper_shared, ballistics_shared, window_shared, shared_exit)
        aim = Process(target=aim_process, args=aim_process_args)
        aim.start()

        render_process_args = (scraper_shared, ballistics_shared, window_shared, shared_exit)
        render = Process(target=render_process, args=render_process_args)
        render.start()

        ctypes.windll.kernel32.SetDllDirectoryW(None)

        while scraper.is_alive() is False or ballistics.is_alive() is False or aim.is_alive() is False or render.is_alive() is False:
            time.sleep(1)

        # if os.path.exists(r'C:/'+loader_name):
        #     os.remove(r'C:/'+loader_name)

        scraper.join()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        error_text = 'error: ' + str(e) + '; type: ' + str(exc_type) + '; file: ' + str(fname) + '; line: ' + str(
            exc_tb.tb_lineno)
        error_log = {'step': 'main', 'error': error_text}
        requests.post(Settings.Product.server + '/error_happen', error_log)

