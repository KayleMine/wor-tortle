import sys
from os import system, getpid
from time import sleep, time
import numpy as np

import ctypes

import psutil

MessageBox = ctypes.windll.user32.MessageBoxW
# Константы для вызова Win32 API
SW_HIDE = 0
SW_SHOW = 5
SW_SHOW_NO_ACTIVE = 4
VK_MENU = 0x12
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x8

# Определение прототипов функций Win32 API
user32 = ctypes.WinDLL('user32.dll')
kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)

# Прототипы функций
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = ctypes.wintypes.HWND

user32.GetWindowTextLengthW.argtypes = [ctypes.wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int

user32.GetClassNameW.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.LPCWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int

kernel32.GetLastError.argtypes = []
kernel32.GetLastError.restype = ctypes.wintypes.DWORD

user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short



def find_window(window_class):
    hwnd = ctypes.windll.user32.FindWindowW(None, None)

    while hwnd != 0:
        class_name = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetClassNameW(hwnd, class_name, ctypes.sizeof(class_name))

        if class_name.value == window_class:
            return hwnd

        hwnd = ctypes.windll.user32.GetWindow(hwnd, ctypes.c_uint(2))

    return False


def show_window(hwnd, show, win_class=None, active=False):
    if show:
        if win_class is not None:
            hwnd = ctypes.windll.user32.FindWindowW(win_class, None)
        result = user32.ShowWindow(hwnd, SW_SHOW if active else SW_SHOW_NO_ACTIVE)
    else:
        result = user32.ShowWindow(hwnd, SW_HIDE)
    return bool(result)


def is_alt_pressed():
    alt_state = user32.GetAsyncKeyState(VK_MENU)

    if alt_state & 0x8000:
        return True
    else:
        return False

def activate_window_by_class(window_class):
    hwnd = ctypes.windll.user32.FindWindowW(window_class, None)
    ctypes.windll.user32.SetForegroundWindow(hwnd)

def get_active_window_class():
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    buff = ctypes.create_unicode_buffer(length + 1)
    user32.GetClassNameW(hwnd, buff, length + 1)
    return buff.value

def remove_topmost_flag(hwnd):
    exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    new_exstyle = exstyle & ~WS_EX_TOPMOST
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_exstyle)


def restore_topmost_flag(hwnd):
    exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    new_exstyle = exstyle | WS_EX_TOPMOST
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_exstyle)

def close_window_by_hwnd(hwnd):
    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)

from functions.processes import get_process

from helpers.fps_manager import FpsManager

from settings import Settings

def window_func(window_shared, shared_lock, shared_exit):
    if Settings.MultiProcessing.is_debug:
        print("[>] Window process pid: " + str(getpid()))

    fps_manager = FpsManager('window', Settings.Window.target_fps)
    prev_window_info = None
    frame_num = 0
    window_info = {
        'window_active': False,
        'process_exist': False
    }
    start_time = time()
    last_process_check = None
    launch_message_box_closed = False
    while shared_exit.is_set() is False:
        try:
            fps_manager.delay()
            frame_num += 1
            active_window_class = ''

            try:
                # window_info['process_exist'] = True
                if last_process_check is None or time() - last_process_check > 30:
                    last_process_check = time()
                    if get_process('aces.exe') is not False:
                        window_info['process_exist'] = True
                    else:
                        window_info['process_exist'] = False
            except:
                pass


            if time() - start_time > 180:
                launch_message_box_closed = True
            if launch_message_box_closed is False:
                launch_message_box = find_window('gui_message_box')
                if launch_message_box:
                    close_window_by_hwnd(launch_message_box)
                    launch_message_box_closed = True


            if (prev_window_info is not None and prev_window_info['process_exist'] is True and window_info['process_exist'] is False) or (window_info['process_exist'] is False and time() - start_time > 200):
                shared_exit.set()
                sys.exit()

            try:
                active_window_class = get_active_window_class()
                window_info['active_class'] = active_window_class
                if active_window_class == 'DagorWClass' or active_window_class == 'GLFW30':
                    window_info['window_active'] = True
                if active_window_class == 'GLFW30':
                    activate_window_by_class('DagorWClass')
                    show_window('', True, win_class='DagorWClass', active=True)
                if Settings.Window.is_debug is True:
                    window_info['window_active'] = True
            except:
                pass

            try:
                if is_alt_pressed():
                    window_info['window_active'] = False
            except:
                pass


            with shared_lock:
                window_shared['value'] = window_info.copy()

            prev_window_info = window_info.copy()

        except KeyboardInterrupt as e:
            shared_exit.set()
            sys.exit()
        except Exception as e:
            if Settings.Memory.is_debug:
                print(e)
            shared_exit.set()
            sys.exit()

