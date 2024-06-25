class Offsets:
    class Base:
        game_ptr = 0x4F50140 + 0x38  # "mem.memreport_sys"
        entity_list_ptr = 0x4FD2AA8 + 0xE0  # "m_player"
        camera_control = 0x4F4E798  #
        entity_list_count = 0x10  # (entity_list_ptr +)
        local_entity_ptr = 0x18  # (entity_list_ptr +)
        game_modes = 0x4EF6A80
        current_game_mode = 0x4F92DC0

    class Game:
        time = 0x1B4
        map_ptr = 0x1D8
        air_units_ptr = 0x2E8
        air_units_count = 0x2F8
        ground_units_ptr = 0x300
        ground_units_count = 0x310
        units_ptr = 0x318
        units_count = 0x328
        ballistics_ptr = 0x408
        camera_ptr = 0x5E8

    class Camera:
        position = 0x58
        view_matrix = 0x1D0

    class Ballistics:
        selected_unit_ptr = 0x668
        weapon_position = 0x1D98
        weapon_position_two = 0x1D6C
        velocity = 0x1D78
        mass = 0x1D84
        caliber = 0x1D88
        length = 0x1D8C
        max_dist = 0x1D90
        bomb_impact_point = 0x1974
        bullet_impact_point = 0x2138
        ingame_ballistics = 0x2190

    class Entity:
        team = 0x210
        gui_state = 0x500
        owned_unit = 0x700

    class Unit:
        byte_size = 0x2500
        bb_min = 0x318
        bb_max = 0x324
        reload_time = 0x978
        rotation = 0xB94 + 0x10
        position = 0xBB8 + 0x10
        air_velocity_ptr = 0xBD8
        air_velocity_offset = 0xB20
        air_acceleration_offset = 0x9A0
        invul_state = 0x1218
        flags = 0x12A8
        is_visible_byte = 0x12A2
        state = 0x12F8
        type = 0x1300
        entity_ptr = 0x1308
        team = 0x1398
        class_ptr = 0x13C0
        unit_info = 0x13A8
        weapon_info = 0x1458
        vehicle_name_ptr = 0x8
        vehicle_class_ptr = 0x38
        # short_vehicle_name_ptr = 0x20
        short_vehicle_name_ptr = 0x8
        ground_velocity = 0x3644
        ground_velocity_ptr = 0x2178
        ground_velocity_offset = 0x54
        damage_model_ptrs = [0x1428, 0x58, 0xA0]
