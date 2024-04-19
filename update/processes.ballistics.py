import os
import sys
from os import getpid
from time import sleep, time
import numpy as np
from math import tan, dist, sqrt

from functions.processes import wait_for_process, get_process

from helpers.fps_manager import FpsManager

from settings import Settings


def ballistics_func(scraper_shared, ballistics_shared, window_shared, shared_lock, shared_exit):
    if Settings.MultiProcessing.is_debug:
        print("[>] Ballistics process pid: " + str(getpid()))

    while get_process(Settings.Memory.process_name) is False:
        sleep(2)
        if shared_exit.is_set():
            sys.exit()

    if Settings.Memory.memory_method != 'kernel':
        if get_process('EasyAntiCheat.exe') is not False or get_process('eac_launcher.exe') is not False:
            shared_exit.set()
            sys.exit()

    fps_manager = FpsManager('ballistics', Settings.Ballistics.target_fps)

    while shared_exit.is_set() is False:
        try:
            fps_manager.delay()
        except KeyboardInterrupt:
            shared_exit.set()
            exit()
        except Exception as e:
            if Settings.Memory.is_debug:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
            continue
