import sys
import os

# Get the current directory of the run.py script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up the directory tree until reaching the "pyp" folder
pyp_path = None
while current_dir != os.path.dirname(current_dir):
    pyp_candidate = os.path.join(current_dir, 'pyp')
    if os.path.exists(pyp_candidate) and os.path.isdir(pyp_candidate):
        pyp_path = pyp_candidate
        break
    current_dir = os.path.dirname(current_dir)


if pyp_path:
    # Append the path to the 'pyp' folder and the 'Lib/site-packages' directory to sys.path
    sys.path.append(os.path.join(pyp_path, 'Lib', 'site-packages'))
else:
    print("Failed to find 'pyp' folder in the directory tree of run.py.")


# Vendors
import requests
from sys import exit
from multiprocessing import freeze_support

import sys
import os
import importlib
import importlib.util

import ctypes
import socket
import hashlib
import platform

embed_path = 'pyp'
loader_name = 'eQDKHsfW82JuNj'#lol

MessageBox = ctypes.windll.user32.MessageBoxW


def load_module_from_file(module_name):
    module_file_path = os.path.join(pyp_path, 'module', f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

def load_modules():
    # Your existing code to fetch module data from the web goes here #pog

    try:
 
        load_module_from_file('offsets')
        load_module_from_file('settings')
        load_module_from_file('usettings')

        load_module_from_file('functions.maths')
        load_module_from_file('functions.ex_ballistics')
        load_module_from_file('functions.esp')
        load_module_from_file('functions.processes')

        load_module_from_file('classes.camera')
        load_module_from_file('classes.entity')
        load_module_from_file('classes.unit')
        load_module_from_file('classes.game')

        load_module_from_file('helpers.memory')
        load_module_from_file('helpers.fps_manager')
        load_module_from_file('helpers.mouse_listener')

        load_module_from_file('processes.window')
        load_module_from_file('processes.scraper')
        load_module_from_file('processes.render')
        load_module_from_file('processes.ballistics')
        load_module_from_file('processes.aim')
        load_module_from_file('processes.main')
    except Exception as e:
        print(e)
        exit()

load_modules()
		
from processes.window import window_func
from processes.scraper import scraper_func
from processes.render import render_func
from processes.ballistics import ballistics_func
from processes.aim import aim_func
from processes.main import main_func
from usettings import UserSettings

def scraper_process(scraper_shared, window_shared, shared_lock, shared_exit):
    scraper_func(scraper_shared, window_shared, shared_lock, shared_exit)


def render_process(scraper_shared, ballistics_shared, window_shared, shared_exit):
    render_func(scraper_shared, ballistics_shared, window_shared, shared_exit)


def ballistics_process(scraper_shared, ballistics_shared, window_shared, shared_lock, shared_exit):
    ballistics_func(scraper_shared, ballistics_shared, window_shared, shared_lock, shared_exit)


def aim_process(scraper_shared, ballistics_shared, window_shared, shared_exit):
    aim_func(scraper_shared, ballistics_shared, window_shared, shared_exit, UserSettings)


def window_process(window_shared, shared_lock, shared_exit):
    window_func(window_shared, shared_lock, shared_exit)


def main():
    main_func(scraper_process, ballistics_process, aim_process, render_process, window_process, embed_path, loader_name)


if __name__ == '__main__':
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        exit()
