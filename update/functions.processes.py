from psutil import process_iter
from time import sleep


def get_process(process_name):
    for process in process_iter(['pid', 'name']):
        if process_name.lower() in process.name().lower():
            return process.pid
    return False


def wait_for_process(process_name, delay=1.0):
    process_pid = False
    while process_pid is False:
        process_pid = get_process(process_name)
        sleep(delay)
