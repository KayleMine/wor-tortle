class Offsets:
    class Base:
        game_ptr = 0x4D79C68 + 0x40  # "mem.memreport_sys"
        entity_list_ptr = 0x4DF4A88 + 0xE0  # "m_player"
        camera_control = 0x4D73468  #
        entity_list_count = 0x10  # (entity_list_ptr +)
        local_entity_ptr = 0x18  # (entity_list_ptr +)
        game_modes = 0x4D217E0
        current_game_mode = 0x4DBFB00

    class Game:
        time = 0x1B4
        map_ptr = 0x1D8
        air_units_ptr = 0x2F8
        air_units_count = 0x308
        ground_units_ptr = 0x310
        ground_units_count = 0x320
        units_ptr = 0x328
        units_count = 0x338
        ballistics_ptr = 0x418
        camera_ptr = 0x608

    class Camera:
        position = 0x68
        view_matrix = 0x1E8

    class Ballistics:
        selected_unit_ptr = 0x628
        weapon_position = 0x1C8C
        weapon_position_two = 0x1CA0
        velocity = 0x1C98
        mass = 0x1CA4
        caliber = 0x1CA8
        length = 0x1CAC
        max_dist = 0x1CB0
        bomb_impact_point = 0x18AC
        bullet_impact_point = 0x2050
        ingame_ballistics = 0x2058

    class Entity:
        team = 0x210
        gui_state = 0x500
        owned_unit = 0x700

    class Unit:
        byte_size = 0x1500
        bb_min = 0x318
        bb_max = 0x324
        reload_time = 0x968
        rotation = 0xB94
        position = 0xBB8
        air_velocity_ptr = 0xBD0
        air_velocity_offset = 0x988
        air_acceleration_offset = 0x9A0
        invul_state = 0x1208
        flags = 0x1298
        is_visible_byte = 0x12A2
        state = 0x12E8
        type = 0x12F0
        entity_ptr = 0x12F8
        team = 0x1380
        class_ptr = 0x13A8
        unit_info = 0x1390
        weapon_info = 0x1440
        vehicle_name_ptr = 0x8
        vehicle_class_ptr = 0x38
        # short_vehicle_name_ptr = 0x20
        short_vehicle_name_ptr = 0x8
        ground_velocity = 0x3644
        damage_model_ptrs = [0x1410, 0x58, 0xA0]
