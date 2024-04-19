import numpy as np
from struct import unpack_from
from offsets import Offsets


class Camera:
    def __init__(self, camera_ptr, memory):
        self.memory = memory

        self.camera_ptr = camera_ptr
        self.bytes = ''
        self.position = np.zeros(3)
        self.view_matrix = np.zeros(16)
        self.current_view = np.zeros(2)

    def update(self):
        self.bytes = self.memory.read_memory(self.camera_ptr, Offsets.Camera.view_matrix + 0x128)
        self.update_position()
        self.update_view_matrix()
        self.update_current_view()
        return self.get_data()

    def update_position(self):
        self.position[:] = unpack_from("<3f", self.bytes, Offsets.Camera.position)
        return self.position

    def update_view_matrix(self):
        self.view_matrix[:] = unpack_from("<16f", self.bytes, Offsets.Camera.view_matrix)
        return self.view_matrix

    def update_current_view(self):
        self.current_view[:] = self.memory.read_float(self.memory.base_address + Offsets.Base.camera_control, 2)

    def set_current_view(self, x, y):
        self.memory.write_float(self.memory.base_address + Offsets.Base.camera_control, x)
        self.memory.write_float(self.memory.base_address + Offsets.Base.camera_control + 0x4, y)

    def get_data(self):
        return {
            'camera_ptr': self.camera_ptr,
            'position': self.position,
            'view_matrix': self.view_matrix,
            'current_view': self.current_view
        }
