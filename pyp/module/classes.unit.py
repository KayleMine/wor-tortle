from time import time
import numpy as np
import os
import sys
from struct import unpack_from
from offsets import Offsets
from settings import Settings


class Unit:
    def __init__(self, unit_ptr, memory):
        self.unit_ptr = unit_ptr
        self.memory = memory
        self.last_read = None
        self.air_velocity_ptr = 0
        self.ground_velocity_ptr = 0
        self.create_time = time()
        self.unit_info_ptr = 0
        self.vehicle_name_ptr = 0
        self.vehicle_class_ptr = 0
        self.bytes = ''
        self.position = np.zeros(3)
        self.prev_position = np.zeros(3)
        self.save_position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.acceleration_alt = np.zeros(3)
        self.average_velocity = np.zeros(3)
        self.prev_velocity = np.zeros(3)
        self.rotation = np.zeros(9)
        self.invul_state = 0
        self.bb_min = np.zeros(3)
        self.bb_max = np.zeros(3)
        self.prev_reload_time = 0
        self.reload_time = 0
        self.max_reload_time = 0
        self.type = -1
        self.team = -1
        self.flags = -1
        self.is_visible_byte = 0
        self.is_visible = False
        self.last_visible = time()
        self.prev_flags = -1
        self.state = -1
        self.dist = float('inf')
        self.cached = True
        self.vehicle_name = "unknown"
        self.vehicle_class = "unknown"
        self.vehicle_name_width = 0
        self.entity_ptr = 0
        self.entity_gui_state = -1
        self.history_last_save = None
        self.is_moving = False
        self.history_delay = 0.6
        self.position_history = []
        self.velocity_history = []
        self.acceleration_history = []
        self.history_times = []
        self.visibility_state = 'cached'
        self.delayed_position = np.zeros(3)
        self.delayed_velocity = np.zeros(3)
        self.delayed_acceleration = np.zeros(3)
        self.acceleration_change = np.zeros(3)
        self.delayed_time = 0
        self.prev_time = 0
        self.history_ready = False
        self.update()

    def update(self):
        try:
            if self.last_read is not None and "Dummy" in self.vehicle_name and time() - self.last_read < 0.5:
                return self.get_data()
            self.last_read = time()
            self.bytes = self.memory.read_memory(self.unit_ptr, Offsets.Unit.byte_size)
            self.update_position()
            self.update_rotation()
            self.update_velocity()
            self.update_bb_min_max()
            self.update_team()
            self.update_reload_time()
            self.update_state()
            self.update_invul_state()
            self.update_flags()
            self.update_type()
            self.update_vehicle_name()
            self.update_entity_ptr()
            self.update_entity_gui_state()
            self.update_visibility_state()
            self.save_history()
            return {'status': 'success', 'data': self.get_data()}
        except Exception as e:
            if Settings.Memory.is_debug:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
            return {'status': 'error', 'data': str(e)}

    def update_entity_gui_state(self):
        entity_ptr = self.entity_ptr
        if entity_ptr != 0:
            entity_gui_state = self.memory.read_byte(entity_ptr + Offsets.Entity.gui_state)
            self.entity_gui_state = entity_gui_state
        return self.entity_gui_state

    def update_bb_min_max(self):
        self.bb_min[:] = unpack_from('<3f', self.bytes, Offsets.Unit.bb_min)
        self.bb_max[:] = unpack_from('<3f', self.bytes, Offsets.Unit.bb_max)
        return self.bb_min, self.bb_max

    def update_reload_time(self):
        self.reload_time = unpack_from('<B', self.bytes, Offsets.Unit.reload_time)[0]
        if self.prev_reload_time == 0 or self.reload_time > self.max_reload_time:
            self.max_reload_time = self.reload_time
        self.prev_reload_time = self.reload_time
        return self.reload_time

    def update_position(self):
        self.is_moving = False
        self.position[:] = unpack_from('<3f', self.bytes, Offsets.Unit.position)
        if self.position[0] == 0 and self.position[2] == 0:
            self.cached = True
            if len(self.position_history) > 0:
                self.position[:] = self.position_history[-1]
                return self.position
        else:
            self.cached = False
            if self.position[0] != self.prev_position[0] or self.position[2] != self.prev_position[2]:
                self.is_moving = True
        self.save_position[:] = self.prev_position
        self.prev_position[:] = self.position
        return self.position

    def update_rotation(self):
        self.rotation[:] = unpack_from('<9f', self.bytes, Offsets.Unit.rotation)
        return self.rotation

    def update_invul_state(self):
        self.invul_state = unpack_from('<B', self.bytes, Offsets.Unit.invul_state)[0]
        return self.rotation

    def update_velocity(self):

        # if self.prev_time != 0 and self.prev_time != self.last_read:
        #     self.velocity = (self.position - self.prev_position) / (self.last_read - self.prev_time)
        try:
            if self.type == 0:
                self.air_velocity_ptr = unpack_from('<Q', self.bytes, Offsets.Unit.air_velocity_ptr)[0]
                if self.air_velocity_ptr != 0:
                    self.velocity[:] = self.memory.read_double(self.air_velocity_ptr + Offsets.Unit.air_velocity_offset, 3)
                    self.acceleration[:] = self.memory.read_double(self.air_velocity_ptr + Offsets.Unit.air_acceleration_offset, 3)
                    if np.linalg.norm(self.acceleration) == 0:
                        if np.linalg.norm(self.delayed_velocity) != 0 and (self.last_read - self.delayed_time) != 0 and len(self.velocity_history) >= 10:
                            time_array = np.array(self.history_times)
                            velocity_array = np.array(self.velocity_history)
                            time_mean = np.mean(time_array)
                            time_std = np.std(time_array)
                            time_normalized = (time_array - time_mean) / time_std
                            coefficients = np.polyfit(time_normalized, velocity_array, 2)
                            self.acceleration[:] = coefficients[1] / time_std
            elif self.type == 8 or self.type == -1:
                self.velocity[:] = [0, 0, 0]
                self.acceleration[:] = [0, 0, 0]
            else:
                self.ground_velocity_ptr = unpack_from('<Q', self.bytes, Offsets.Unit.ground_velocity_ptr)[0]
                # self.velocity[:] = unpack_from('<3f', self.bytes, Offsets.Unit.ground_velocity)
                self.velocity[:] = self.memory.read_float(self.ground_velocity_ptr + Offsets.Unit.ground_velocity_offset, 3)
            if self.velocity[0] != self.prev_velocity[0] or self.velocity[2] != self.prev_velocity[2]:
                self.prev_velocity[:] = self.velocity
                self.prev_time = self.last_read

        except Exception as e:
            print(e)
            self.velocity[:] = [0, 0, 0]
            self.acceleration[:] = [0, 0, 0]

        return self.velocity

    def update_type(self):
        self.type = unpack_from('<B', self.bytes, Offsets.Unit.type)[0]
        return self.type

    def update_team(self):
        self.team = unpack_from('<B', self.bytes, Offsets.Unit.team)[0]
        return self.team

    def update_state(self):
        self.state = unpack_from('<B', self.bytes, Offsets.Unit.state)[0]
        return self.state

    def update_flags(self):
        self.flags = unpack_from('<I', self.bytes, Offsets.Unit.flags)[0]
        self.is_visible_byte = unpack_from('<B', self.bytes, Offsets.Unit.is_visible_byte)[0]
        return self.flags

    def update_visibility_state(self):

        # if (self.flags & 0x800) and (self.flags & 0x40000000 == 0 or ((self.flags & 0x2000) and (self.flags & 0x8))):
        if (self.flags & 0x800):
            state = 'truly_visible'
        elif (self.flags & 0x800 and self.flags & 0x40000000):
            state = 'known'
        else:
            state = 'invisible'


        if self.cached:
            state = 'cached'
        self.visibility_state = state
        return state
        flag_hide = 0x80000000
        flag_visible = 0x800
        flag_truly_visible = 0x80000
        flag_known = 0x1000
        flag_scouted = 0x40000000
        flat_not_loaded = 0x4
        return self.visibility_state

    def update_entity_ptr(self):
        self.entity_ptr = unpack_from('<Q', self.bytes, Offsets.Unit.entity_ptr)[0]
        return self.entity_ptr

    def update_vehicle_name(self):
        self.unit_info_ptr = int(unpack_from('<Q', self.bytes, Offsets.Unit.unit_info)[0])
        if self.unit_info_ptr != 0:
            self.vehicle_name_ptr = int(self.memory.read_int8(self.unit_info_ptr + Offsets.Unit.short_vehicle_name_ptr)[0])
            if self.vehicle_name_ptr != 0:
                try:
                    self.vehicle_name = self.memory.read_text(self.vehicle_name_ptr, 25).replace("_", " ").capitalize()
                except Exception as e:
                    self.vehicle_name = 'unknown'
                if 'Unit hangar' in self.vehicle_name:
                    self.vehicle_name = 'WC Enjoyer'

            self.vehicle_class_ptr = int(
                self.memory.read_int8(self.unit_info_ptr + Offsets.Unit.vehicle_class_ptr)[0])
            if self.vehicle_class_ptr != 0:
                self.vehicle_class = self.memory.read_text(self.vehicle_class_ptr, 25)

        return self.vehicle_name

    def save_history(self):
        if self.entity_gui_state == 8:
            self.position_history = []
            self.velocity_history = []
            self.history_times = []
            return False
        if self.cached is False and self.is_moving:
            dt = self.last_read - self.delayed_time
            if dt > 0 and self.delayed_time != 0:
                self.acceleration_change[:] = (self.acceleration - self.delayed_acceleration) / dt
            self.position_history.append(self.position.copy())
            self.velocity_history.append(self.velocity.copy())
            self.acceleration_history.append(self.acceleration.copy())
            self.history_times.append(self.last_read)
            if time() - self.history_times[0] >= self.history_delay:
                self.history_ready = True
                self.position_history.pop(0)
                self.velocity_history.pop(0)
                self.acceleration_history.pop(0)
                self.history_times.pop(0)
            if len(self.velocity_history) > 1:
                self.average_velocity[:] = np.sum(self.velocity_history, axis=0) / len(self.velocity_history)
            self.delayed_position[:] = self.position_history[0]
            self.delayed_velocity[:] = self.velocity_history[0]
            self.delayed_acceleration[:] = self.acceleration_history[0]
            self.delayed_time = self.history_times[0]
        return self.position_history

    def get_data(self):
        return {
            'vehicle_name': self.vehicle_name,
            'read_time': self.last_read,
            'unit_ptr': self.unit_ptr,
            'entity_ptr': self.entity_ptr,
            'entity_gui_state': self.entity_gui_state,
            'position': self.position,
            'prev_position': self.save_position,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'acceleration_alt': self.acceleration_alt,
            'average_velocity': self.average_velocity,
            'prev_velocity': self.prev_velocity,
            'dist': round(self.dist, 3),
            'vehicle_name_width': self.vehicle_name_width,
            'vehicle_class': self.vehicle_class,
            'team': self.team,
            'state': self.state,
            'reload_time': self.reload_time,
            'max_reload_time': self.max_reload_time,
            'type': self.type,
            'invul_state': self.invul_state,
            'flags': self.flags,
            'cached': self.cached,
            'bb_min': self.bb_min,
            'bb_max': self.bb_max,
            'is_moving': self.is_moving,
            'visibility_state': self.visibility_state,
            'delayed_position': self.delayed_position,
            'delayed_velocity': self.delayed_velocity,
            'delayed_acceleration': self.delayed_acceleration,
            'acceleration_change': self.acceleration_change,
            'delayed_time': self.delayed_time,
            'history_ready': self.history_ready
        }