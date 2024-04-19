from struct import unpack_from
from offsets import Offsets


class Entity:
    def __init__(self, entity_ptr, memory):
        self.memory = memory

        self.entity_ptr = entity_ptr
        self.unit_ptr = 0
        self.bytes = ''
        self.team = -1
        self.gui_state = -1
        self.update()

    def update(self):
        try:
            self.bytes = self.memory.read_memory(self.entity_ptr, Offsets.Entity.owned_unit + 0x8)
            self.update_team()
            self.update_gui_state()
            self.update_unit_ptr()
            return {'status': 'success', 'data': self.get_data()}
        except Exception as e:
            return {'status': 'error', 'data': str(e)}

    def update_team(self):
        self.team = unpack_from('<B', self.bytes, Offsets.Entity.team)[0]
        return self.team

    def update_gui_state(self):
        self.gui_state = unpack_from('<B', self.bytes, Offsets.Entity.gui_state)[0]
        return self.gui_state

    def update_unit_ptr(self):
        self.unit_ptr = unpack_from('<Q', self.bytes, Offsets.Entity.owned_unit)[0]
        return self.unit_ptr

    def get_data(self):
        return {
            'entity_ptr': self.entity_ptr,
            'unit_ptr': self.unit_ptr,
            'team': self.team,
            'gui_state': self.gui_state
        }