# Vendors
from os import system, getpid
from sys import exit
import sys
import os
from time import sleep, time

from functions.processes import get_process

from helpers.memory import Memory
from helpers.fps_manager import FpsManager

from classes.game import Game

from settings import Settings


def scraper_func(scraper_shared, window_shared, shared_lock, shared_exit):
    if Settings.MultiProcessing.is_debug:
        print("[>] Scraper process pid: " + str(getpid()))

    while get_process(Settings.Memory.process_name) is False:
        sleep(2)
        if shared_exit.is_set():
            sys.exit()
    if Settings.Memory.memory_method != 'kernel':
        if get_process('EasyAntiCheat.exe') is not False or get_process('eac_launcher.exe') is not False:
            print('')
            print("[X] ERROR: [ARCADE] not support EasyAntiCheat. Please, disable EasyAntiCheat or "
                  "use [Kernel] or [Exclusive] version")
            shared_exit.set()
            sys.exit()

    sleep(1)

    memory = Memory(Settings.Memory.process_name, Settings.Memory.memory_method)
    if memory.ready is False:
        shared_exit.set()
        sys.exit()
    game = Game(memory)
    fps_manager = FpsManager('scraper', Settings.Scraper.target_fps)
    while shared_exit.is_set() is False:
        try:
            fps_manager.delay()

            window_info = window_shared.copy()['value']
            if window_info.get('window_active') is not True:
                continue

            scraped_info = game.update()

            if scraped_info['status'] == 'skip' or scraped_info['status'] == 'error':
                if get_process(Settings.Memory.process_name) is False:
                    shared_exit.set()
                    exit()
                if Settings.Scraper.is_debug:
                    print(scraped_info)
                game.clear_cache()
                scraped_info['data'] = {'camera_ptr': None, 'local_unit': None, 'enemy_units': {}, 'entity_list_count': 0, 'all_units_ptrs':[], 'perform_units': [], 'skip_unit_ptrs': []}
                sleep(0.2)

            with shared_lock:
                scraper_shared['value'] = scraped_info['data']

        except KeyboardInterrupt:
            shared_exit.set()
            exit()

        except Exception as e:
            if Settings.Memory.is_debug:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
            if get_process('aces.exe') is False:
                shared_exit.set()
                sys.exit()
            continue