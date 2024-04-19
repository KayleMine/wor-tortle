import sys
import os
import numpy as np
from math import dist
from struct import unpack_from
from offsets import Offsets
from classes.unit import Unit
from classes.entity import Entity

from settings import Settings

class Game:
    def __init__(self, memory):
        self.memory = memory

        self.game_ptr = 0
        self.game_mode = ''
        self.game_mode_id = 0
        self.game_mode_ptr = 0

        self.bytes = ''

        self.camera_ptr = 0
        self.ballistics_ptr = 0

        self.all_units_ptr = 0
        self.all_units_count = 0
        self.all_units_ptrs = []

        self.air_units_ptr = 0
        self.air_units_count = 0
        self.air_units_ptrs = []

        self.ground_units_ptr = 0
        self.ground_units_count = 0
        self.ground_units_ptrs = []

        self.entity_list_ptr = 0
        self.entity_list_count = 0
        self.entity_list = []
        self.skip_entity = []

        self.local_entity_ptr = 0
        self.prev_local_entity_ptr = 0
        self.is_new_game = True

        self.skip_unit_ptrs = []
        self.units_ptrs = []

        self.local_entity = None
        self.local_unit = None
        self.units = {}
        self.enemy_units = {}
        self.count_by_class = {}
        self.perform_units = []

        self.bomb_impact_point = np.zeros(3)

        self.camera_vector = np.zeros(3)

        self.ballistics_bytes = ''
        self.weapon_position = np.zeros(3)
        self.weapon_position_alt = np.zeros(3)
        self.weapon = {'ready': False, 'velocity': 0, 'mass': 0, 'caliber': 0, 'length': 0, 'weapon_position': [0, 0, 0]}

    def update(self):
        try:
            self.update_game_ptr()
            if self.game_ptr == 0:
                return {'status': 'skip', 'data': 'empty_game_ptr'}

            self.bytes = self.memory.read_memory(self.game_ptr, Offsets.Game.camera_ptr + 0x8)

            self.update_camera_ptr()
            if self.camera_ptr == 0:
                return {'status': 'skip', 'data': 'empty_camera_ptr'}

            self.update_entity_list()
            if self.entity_list_count == 0:
                return {'status': 'skip', 'data': 'empty_entity_list'}

            if self.entity_list_count == 1:
                self.update_air_units_ptr()
                self.update_all_units_ptr()
                if self.all_units_count == 0:
                    return {'status': 'skip', 'data': 'empty_units'}

            self.update_local_entity()
            if self.local_entity is None or self.local_entity.unit_ptr == 0:
                self.local_entity = None
                return {'status': 'skip', 'data': 'empty_local_entity'}

            self.update_local_unit()

            self.update_units_ptrs()

            self.update_enemy_units()

            self.update_ballistics_info()

            return {'status': 'success', 'data': self.get_data()}
        except Exception as e:
            if Settings.Memory.is_debug:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
            return {'status': 'error', 'data': str(e)}

    def update_game_ptr(self):
        if self.game_ptr == 0:
            self.game_ptr = self.memory.read_int8(self.memory.base_address + Offsets.Base.game_ptr)[0]

        try:
            self.game_mode_id = self.memory.read_int4(self.memory.base_address + Offsets.Base.current_game_mode)
            self.game_mode_ptr = self.memory.read_int8(self.memory.base_address + Offsets.Base.game_modes + (0x8 * self.game_mode_id))[0]
            if self.game_mode_ptr != 0:
                self.game_mode = self.memory.read_text(self.game_mode_ptr, 25)
        except:
            self.game_mode = 'loading'

        return self.game_ptr

    def update_bomb_impact_point(self):
        if self.ballistics_ptr != 0:
            self.bomb_impact_point[:] = unpack_from('<3f', self.ballistics_bytes, Offsets.Ballistics.bomb_impact_point)
        else:
            self.bomb_impact_point[:] = [0, 0, 0]


    def update_camera_ptr(self):
        if self.camera_ptr == 0:
            self.camera_ptr = unpack_from('<Q', self.bytes, Offsets.Game.camera_ptr)[0]
        return self.camera_ptr


    def update_air_units_ptr(self):
        self.air_units_ptr = unpack_from('<Q', self.bytes, Offsets.Game.air_units_ptr)[0]
        self.air_units_count = unpack_from('<I', self.bytes, Offsets.Game.air_units_count)[0]
        if self.air_units_count > 500:
            self.air_units_count = 500
        return self.air_units_ptr

    def update_ground_units_ptr(self):
        self.ground_units_ptr = unpack_from('<Q', self.bytes, Offsets.Game.ground_units_ptr)[0]
        self.ground_units_count = unpack_from('<I', self.bytes, Offsets.Game.ground_units_count)[0]
        if self.ground_units_count > 500:
            self.ground_units_count = 500
        return self.ground_units_ptr

    def update_all_units_ptr(self):
        self.all_units_ptr = unpack_from('<Q', self.bytes, Offsets.Game.units_ptr)[0]
        self.all_units_count = unpack_from('<I', self.bytes, Offsets.Game.units_count)[0]
        if self.all_units_count > 500:
            self.all_units_count = 500
        return self.all_units_ptr

    def update_local_entity(self):
        if self.local_entity is None:
            self.local_entity = Entity(self.local_entity_ptr, self.memory)
        else:
            self.local_entity.update()
        return self.local_entity

    def update_local_unit(self):
        if self.local_unit is None:
            self.local_unit = Unit(self.local_entity.unit_ptr, self.memory)
        else:
            self.local_unit.update()

    def update_is_new_game(self):
        if self.local_entity_ptr != self.prev_local_entity_ptr:
            self.is_new_game = True
        else:
            self.is_new_game = False
        self.prev_local_entity_ptr = self.local_entity_ptr
        if self.is_new_game:
            self.clear_cache()
        return self.is_new_game

    def clear_cache(self):
        self.skip_unit_ptrs = []
        self.units = {}
        self.local_unit = None
        self.local_entity = None

    def update_entity_list(self):
        self.update_entity_list_ptr()
        if self.entity_list_count <= 1:
            self.entity_list = []
            self.skip_entity = []
            return False

    def update_entity_list_ptr(self):
        entity_list_bytes = self.memory.read_memory(self.memory.base_address + Offsets.Base.entity_list_ptr,
                                                    Offsets.Base.local_entity_ptr + 0x8)
        self.entity_list_ptr = unpack_from('<Q', entity_list_bytes)[0]
        self.entity_list_count = unpack_from('<I', entity_list_bytes, Offsets.Base.entity_list_count)[0]
        if self.entity_list_count > 500:
            self.entity_list_count = 500
        self.local_entity_ptr = unpack_from('<Q', entity_list_bytes, Offsets.Base.local_entity_ptr)[0]
        del entity_list_bytes
        self.update_is_new_game()
        return self.entity_list_ptr

    def update_units_ptrs(self):
        if self.entity_list_count == 1:
            self.all_units_ptrs = unpack_from("<" + str(self.all_units_count) + "Q",
                                         self.memory.read_memory(self.all_units_ptr,
                                                                 self.all_units_count * 0x8))
        else:
            self.entity_list = self.memory.read_int8(self.entity_list_ptr, self.entity_list_count)
            perform_entities = [x for x in self.entity_list if x not in self.skip_entity]
            enemy_units = []
            for entity_ptr in perform_entities:
                # entity_team = int(self.memory.read_byte(entity_ptr + Offsets.Entity.team))
                # if entity_team == self.local_entity.team and entity_team != 0:
                #     self.skip_entity.append(entity_ptr)
                #     continue
                enemy_units.append(self.memory.read_int8(entity_ptr + Offsets.Entity.owned_unit)[0])
            # if self.air_units_ptr != 0:
            #     air_units = self.memory.read_int8(self.air_units_ptr, self.air_units_count)
            #     enemy_units = list(enemy_units + air_units)
            self.all_units_ptrs = enemy_units

        return {'all_units_ptrs': self.all_units_ptrs}

    def update_enemy_units(self):
        all_units_ptrs = self.all_units_ptrs
        skip_unit_ptrs = self.skip_unit_ptrs
        local_entity = self.local_entity
        local_unit = self.local_unit
        units = self.units

        skip_unit_ptrs = [x for x in skip_unit_ptrs if x in all_units_ptrs]
        perform_units = [x for x in all_units_ptrs if x not in skip_unit_ptrs]

        self.count_by_class = {}
        enemy_units = {}
        for unit_ptr in perform_units:
            if unit_ptr == 0:
                skip_unit_ptrs.append(unit_ptr)
                continue

            unit = units.get(unit_ptr)
            if unit is None:
                unit = Unit(unit_ptr, self.memory)
                units[unit_ptr] = unit
            else:
                unit.update()

            # skip for LOOP | Not loaded
            if 0x4 & unit.flags:
                continue

            if unit.entity_ptr == local_entity.entity_ptr:
                skip_unit_ptrs.append(unit_ptr)
                continue

            # skip for GAME | Teammate
            if unit.team == local_entity.team and unit.team > 0 and Settings.Scraper.is_death_match is False:
                skip_unit_ptrs.append(unit_ptr)
                continue

            # skip for GAME | Decor and map objects
            if ((unit.type == 8 or unit.type == 4) and "Bullshit Ware" not in unit.vehicle_name):
                # skip_unit_ptrs.append(unit_ptr)
                continue

            # skip for GAME | Dead unit
            if (unit.state == 2 or unit.state == 3) and (
                    unit.entity_gui_state != 0 and unit.entity_gui_state != 1 and unit.entity_gui_state != 2) and self.entity_list_count > 1:
                # skip_unit_ptrs.append(unit_ptr)
                continue

            # skip for LOOP | Dead, but maybe not loaded
            if unit.state == 2 or unit.state == 4:
                continue

            # skip for LOOP | Decor, bot or temp dummy
            if unit.entity_ptr == 0 and self.entity_list_count > 1:
                continue

            # skip for LOOP | Cached
            if unit.position[0] == 0 and unit.position[2] == 0:
                continue

            # skip for LOOP | Temp dummy
            if "Dummy" in unit.vehicle_name:
                continue

            if self.count_by_class.get(unit.vehicle_class) is not None:
                self.count_by_class[unit.vehicle_class] += 1
            else:
                self.count_by_class[unit.vehicle_class] = 1

            unit.dist = dist(local_unit.position, unit.position)

            enemy_units[unit_ptr] = unit.get_data()

        enemy_units = dict(sorted(enemy_units.items(), key=lambda x: x[1]['dist'], reverse=True))

        self.perform_units = perform_units
        self.enemy_units = enemy_units
        self.skip_unit_ptrs = skip_unit_ptrs


    def update_ballistics_info(self):
        try:
            self.ballistics_ptr = unpack_from('<Q', self.bytes, Offsets.Game.ballistics_ptr)[0]
            if self.ballistics_ptr == 0:
                return {'ready': False, 'velocity': 0, 'mass': 0, 'caliber': 0, 'length': 0}

            self.ballistics_bytes = self.memory.read_memory(self.ballistics_ptr, Offsets.Ballistics.ingame_ballistics + 0x18)
            if self.ballistics_bytes == "":
                return {'ready': False, 'velocity': 0, 'mass': 0, 'caliber': 0, 'length': 0, 'in_game': [0,0,0]}

            max_dist = unpack_from('<f', self.ballistics_bytes, Offsets.Ballistics.max_dist)[0]
            velocity = unpack_from('<f', self.ballistics_bytes, Offsets.Ballistics.velocity)[0]
            mass = unpack_from('<f', self.ballistics_bytes, Offsets.Ballistics.mass)[0]
            caliber = unpack_from('<f', self.ballistics_bytes, Offsets.Ballistics.caliber)[0]
            length = unpack_from('<f', self.ballistics_bytes, Offsets.Ballistics.length)[0]
            selected_unit = unpack_from('<Q', self.ballistics_bytes, Offsets.Ballistics.selected_unit_ptr)[0]
            ingame_result = unpack_from('<3f', self.ballistics_bytes, Offsets.Ballistics.ingame_ballistics)
            self.weapon_position[:] = unpack_from('<3f', self.ballistics_bytes, Offsets.Ballistics.weapon_position)
            crosshair_info_ptr = unpack_from('<Q', self.ballistics_bytes, 0xA70)[0]
            if crosshair_info_ptr != 0:
                crosshair_position_ptr = self.memory.read_int8(crosshair_info_ptr + 0x628)[0]
                if crosshair_position_ptr != 0:
                    # self.weapon_position_alt[:] = self.memory.read_float(crosshair_position_ptr + 0x688, 3)
                    self.weapon_position_alt[:] = self.memory.read_float(crosshair_position_ptr + 0x5FC, 3)
            # self.weapon_position_alt[:] = unpack_from('<3f', self.ballistics_bytes, Offsets.Ballistics.weapon_position_two)

            if mass > 0 and caliber > 0 and length > 0 and velocity > 0:
                self.weapon = {'ready': True, 'velocity': velocity, 'mass': mass, 'caliber': caliber, 'length': length, 'in_game': ingame_result, 'selected_unit': selected_unit, 'max_dist': max_dist, 'weapon_position': self.weapon_position,'weapon_position_alt': self.weapon_position_alt}
            else:
                self.weapon = {'ready': False, 'velocity': 0, 'mass': 0, 'caliber': 0, 'length': 0, 'in_game': [0,0,0], 'weapon_position': [0, 0, 0], 'weapon_position_alt': [0, 0, 0]}

            if self.local_unit is not None and self.local_unit.type == 0:
                self.update_bomb_impact_point()
            else:
                self.bomb_impact_point[:] = [0, 0, 0]

        except Exception as e:
            if Settings.Memory.is_debug:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


    def get_data(self):
        return {
            'game_mode': self.game_mode,
            'game_ptr': self.game_ptr,
            'camera_ptr': self.camera_ptr,
            'local_unit': self.local_unit.get_data(),
            'enemy_units': self.enemy_units,
            'weapon': self.weapon,
            'entity_list_count': self.entity_list_count,
            'all_units_ptrs': self.all_units_ptrs,
            'perform_units': self.perform_units,
            'skip_unit_ptrs': self.skip_unit_ptrs,
            'bomb_impact_point': self.bomb_impact_point,
            'camera_vector': self.camera_vector,
            'count_by_class': self.count_by_class,
        }
