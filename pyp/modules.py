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
    
update_path = None
while current_dir != os.path.dirname(current_dir):
    update_candidate = os.path.join(current_dir, 'update')
    if os.path.exists(update_candidate) and os.path.isdir(update_candidate):
        update_path = update_candidate
        break
    current_dir = os.path.dirname(current_dir)


if pyp_path:
    # Append the path to the 'pyp' folder and the 'Lib/site-packages' directory to sys.path
    sys.path.append(os.path.join(pyp_path, 'Lib', 'site-packages'))
else:
    print("Failed to find 'pyp' folder in the directory tree of run.py.")


import requests
import json
import importlib.util
import sys
import base64

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
def encrypt(string: str, key: str) -> str:
    data = string.encode()
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    iv = base64.b64encode(cipher.iv).decode('utf-8')
    return f"{ct}__{iv}"


def decrypt(encrypted: str, key: str) -> str:
    ct, iv = encrypted.split('__')
    iv = base64.b64decode(iv)
    ct = base64.b64decode(ct)
    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode()


def load_module_from_string(module_code, module_name):
    module_spec = importlib.util.spec_from_loader(module_name, loader=None, origin='string', is_package=False)
    module = importlib.util.module_from_spec(module_spec)
    exec(module_code, module.__dict__)
    sys.modules[module_name] = module


import os

def load_module(path, modules_data, module_name):
    module_code = modules_data[module_name]
    module_file_name = os.path.join(path, f"{module_name}.py")
    with open(module_file_name, "w") as module_file:
        module_file.write(module_code)


# Customs
def load_modules():
    try:
        modules_req = requests.post('https://warchill.xyz/get_modules', {'s': '7GBT3-WNXM9-MZ3RW-6VF3Y', 'hwid': '0322-7015-3f5c-aa89'})
        modules_answer = json.loads(modules_req.text)
    except Exception as e:
        exit()

    if modules_answer.get('status') == 'success':
        modules_data = json.loads(
            decrypt(modules_answer.get('modules'), base64.b64decode('BLz29I71cjjEEbl6L3hEw01ys/43cdI0vG0ZHekaFj8=')))
    else:
        exit()

    try:
        if update_path:
            load_module(update_path, modules_data, 'offsets')
            load_module(update_path, modules_data, 'settings')
            load_module(update_path, modules_data, 'usettings')

            load_module(update_path, modules_data, 'functions.maths')
            load_module(update_path, modules_data, 'functions.ballistics')
            load_module(update_path, modules_data, 'functions.esp')
            load_module(update_path, modules_data, 'functions.processes')

            load_module(update_path, modules_data, 'classes.camera')
            load_module(update_path, modules_data, 'classes.entity')
            load_module(update_path, modules_data, 'classes.unit')
            load_module(update_path, modules_data, 'classes.game')

            load_module(update_path, modules_data, 'helpers.memory')
            load_module(update_path, modules_data, 'helpers.fps_manager')
            load_module(update_path, modules_data, 'helpers.mouse_listener')

            load_module(update_path, modules_data, 'processes.window')
            load_module(update_path, modules_data, 'processes.scraper')
            load_module(update_path, modules_data, 'processes.render')
            load_module(update_path, modules_data, 'processes.ballistics')
            load_module(update_path, modules_data, 'processes.aim')
            load_module(update_path, modules_data, 'processes.main')
            print("Now move files from update folder to pyp\module")
        else:
            print("Failed to find 'update' folder in the directory tree of modules.py.")


    except Exception as e:
        print(e)
        exit()

load_modules()