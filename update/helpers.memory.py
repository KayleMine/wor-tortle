import pymem
import ctypes


from time import sleep as wait
from os import system, getpid
from sys import exit
from struct import unpack, pack_into

from functions.processes import get_process, wait_for_process


class Memory:
    def __init__(self, process_name, method='user'):
        self.method = method
        self.process_name = process_name
        self.process_handle = False
        self.ready = False
        self.process_pid = get_process(process_name)
        if self.process_pid is False:
            print('[X] ERROR: Target process is not exist')
        else:
            try:
                self.process_handle = pymem.Pymem(self.process_name)
                self.base_address = self.get_base_address()
                self.ready = True
            except Exception as e:
                print('[X] ERROR: Memory driver | ' + str(e))

    def get_base_address(self):
        return self.process_handle.base_address

    def read_memory(self, address, size):
        return self.process_handle.read_bytes(address, size)

    def read_int4(self, address):
        read_bytes = self.read_memory(address, 4)
        result = unpack('<I', read_bytes)[0]
        return result

    def read_int8(self, address, count=1):
        read_bytes = self.read_memory(address, count * 8)
        result = unpack('<' + str(count) + 'Q', read_bytes)
        return result

    def read_text(self, address, tlength=50):
        read_bytes = self.read_memory(address, tlength)
        size = 0
        for byte in read_bytes:
            if byte == 0:
                break
            size += 1
        data = read_bytes[:size]
        return data.decode('utf-8', errors='replace')

    def read_pointer(self, address):
        read_bytes = self.read_memory(address, 8)
        result = unpack('<P', read_bytes)[0]
        return result

    def read_float(self, address, count=1):
        data = self.read_memory(address, 0x4 * count)
        result = unpack(str(count) + 'f', data)
        return result

    def read_double(self, address, count=1):
        data = self.read_memory(address, 0x8 * count)
        result = unpack(str(count) + 'd', data)
        return result

    def read_byte(self, address):
        data = self.read_memory(address, 1)
        result = int.from_bytes(data, byteorder='big')
        return result

    def write_float(self, address, value):
        return self.process_handle.write_float(address, value)

    def write_int(self, address, val):
        return self.process_handle.write_int(address, val)

    def write_memory(self, address, data):
        return self.process_handle.write_longlong(address, data)

    def write_byte(self, address, value):
       return self.process_handle.write_int(address, value)


