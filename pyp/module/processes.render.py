import os
from os import system, getpid
from sys import exit
from time import sleep, time
from math import sqrt, dist, tan, acos, degrees, floor, ceil
import random
import string
import ctypes
import numpy as np
import sys

import raypyc
import raypyc as render_engine

from functions.processes import get_process
from functions.maths import w2s, get_direct_angle, rotate_points, from_axis_angle, transform
from functions.esp import draw_text, draw_3d_box_smart, draw_debug_list, get_box_color, draw_enemy_info

from helpers.memory import Memory
from helpers.fps_manager import FpsManager

from classes.camera import Camera

from offsets import Offsets
from settings import Settings

SW_HIDE = 0
SW_SHOW = 5
SW_SHOW_NO_ACTIVE = 4
GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x8
user32 = ctypes.WinDLL('user32.dll')

user32.ShowWindow.argtypes = [ctypes.wintypes.HWND, ctypes.c_int]
user32.ShowWindow.restype = ctypes.wintypes.BOOL


def show_window(hwnd, show, win_class=None, active=False):
    if show:
        if win_class is not None:
            hwnd = ctypes.windll.user32.FindWindowW(win_class, None)
        result = user32.ShowWindow(hwnd, SW_SHOW if active else SW_SHOW_NO_ACTIVE)
    else:
        result = user32.ShowWindow(hwnd, SW_HIDE)
    return bool(result)


def remove_topmost_flag(hwnd):
    exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    new_exstyle = exstyle & ~WS_EX_TOPMOST
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_exstyle)


def restore_topmost_flag(hwnd):
    exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    new_exstyle = exstyle | WS_EX_TOPMOST
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_exstyle)


def activate_window_by_class(window_class):
    hwnd = ctypes.windll.user32.FindWindowW(window_class, None)
    ctypes.windll.user32.SetForegroundWindow(hwnd)


def calculate_correction(camera_coords, gun_coords, target_coords, convergence_point=1000):
    # Calculate the vector from the camera to the gun and normalize it
    camera_to_gun = gun_coords - camera_coords
    camera_to_gun /= np.linalg.norm(camera_to_gun)
    # Calculate the vector from the camera to the target
    camera_to_target = target_coords - camera_coords
    # Calculate the distance from the camera to the target
    distance_to_target = np.linalg.norm(camera_to_target)
    # Calculate the correction
    correction = camera_to_target - camera_to_gun * distance_to_target
    return correction

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotate_vector(vector, angle, axis):

    # Ensure that the input vector is a 2D array
    vector = np.atleast_2d(vector)

    # Normalize the axis of rotation
    axis = axis / np.linalg.norm(axis)

    # Create a rotation matrix for the given angle(s) and axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle + axis[0] ** 2 * (1 - cos_angle),
         axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
         axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle],
        [axis[0] * axis[1] * (1 - cos_angle) + axis[2] * sin_angle,
         cos_angle + axis[1] ** 2 * (1 - cos_angle),
         axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle],
        [axis[0] * axis[2] * (1 - cos_angle) - axis[1] * sin_angle,
         axis[1] * axis[2] * (1 - cos_angle) + axis[0] * sin_angle,
         cos_angle + axis[2] ** 2 * (1 - cos_angle)]
    ])

    # Apply the rotation matrix to the vector(s)
    rotated_vector = np.dot(vector, rotation_matrix.T)

    # If the input was a single vector, return a 1D array
    if rotated_vector.shape == (1, 3):
        rotated_vector = rotated_vector.squeeze()

    return rotated_vector





def render_func(scraper_shared, ballistics_shared, window_shared, shared_exit):
    if Settings.MultiProcessing.is_debug:
        print("[>] Render process pid: " + str(getpid()))

    try:
        overlay_title = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        screen_w, screen_h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        screen_w -= 1
        screen_h -= 1
        render_engine.set_trace_log_level(5)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_WINDOW_TRANSPARENT)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_WINDOW_HIGHDPI)
        render_engine.set_target_fps(Settings.Render.target_fps)
        if Settings.Render.enable_msaa_x4:
            render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_MSAA_4X_HINT)
        if Settings.Render.enable_vsync:
            render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_VSYNC_HINT)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_INTERLACED_HINT)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_WINDOW_TOPMOST)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_WINDOW_ALWAYS_RUN)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_WINDOW_UNDECORATED)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_WINDOW_MOUSE_PASSTHROUGH)
        render_engine.set_config_flags(render_engine.ConfigFlags.FLAG_WINDOW_UNFOCUSED)
        render_engine.init_window(screen_w, screen_h, overlay_title.encode())
        render_engine.set_window_position(0, 0)
        overlay_handle = render_engine.get_window_handle()
        show_window(overlay_handle, False)
    except Exception as e:
        MessageBox = ctypes.windll.user32.MessageBoxW
        MessageBox(None, '[Error] Cant create overlay: ' + str(e), 'Error', 0)
        sys.exit()

    while get_process(Settings.Memory.process_name) is False:
        render_engine.begin_drawing()
        render_engine.clear_background(render_engine.BLANK)
        render_engine.end_drawing()
        sleep(2)
        if shared_exit.is_set():
            sys.exit()

    if Settings.Memory.memory_method != 'kernel':
        if get_process('EasyAntiCheat.exe') is not False or get_process('eac_launcher.exe') is not False:
            print('')
            print("[X] ERROR: [ARCADE] not support EasyAntiCheat. Please, disable EasyAntiCheat or "
                  "use [Kernel] or [Exclusive] version")
            shared_exit.set()
            sys.exit()

    scraped_info = scraper_shared['value'].copy()
    while scraped_info.get('camera_ptr') is None or scraped_info.get('camera_ptr') == 0:
        try:
            render_engine.begin_drawing()
            render_engine.clear_background(render_engine.BLANK)
            render_engine.end_drawing()
            scraped_info = scraper_shared.copy()['value']
            sleep(0.5)
        except KeyboardInterrupt:
            sys.exit()

    camera_ptr = scraped_info.get('camera_ptr')
    sleep(2)
    memory = Memory(Settings.Memory.process_name, Settings.Memory.memory_method)
    if memory.ready is False:
        shared_exit.set()
        sys.exit()

    predicted_position = np.zeros(3)
    weapon_position = np.zeros(3)
    weapon_vector = np.zeros(3)
    local_position = np.zeros(3)
    local_velocity = np.zeros(3)
    local_acceleration = np.zeros(3)
    impact_point = np.zeros(3)
    view_matrix = np.zeros(16)
    camera_position = np.zeros(3)
    ingame_crosshair_position = np.zeros(3)
    bomb_impact_point = np.zeros(3)
    bomb_impact_size = np.zeros(3)
    velocity_towards_target = np.zeros(3)
    acceleration_towards_target = np.zeros(3)
    text_position = np.zeros(2)
    final_predicted_position = np.zeros(3)
    custom_crosshair_position = np.zeros(3)
    reload_text_width = render_engine.measure_text('Reloading'.encode(), 10)
    prev_window_info = None
    while shared_exit.is_set() is False:
        if shared_exit.is_set():
            sys.exit()

        try:
            scraped_info = scraper_shared.copy()['value']
            ballistics_info = ballistics_shared.copy()['value']
            window_info = window_shared.copy()['value']

            enemies = scraped_info['enemy_units']
            local_unit = scraped_info.get('local_unit')

            view_matrix[:] = memory.read_float(camera_ptr + Offsets.Camera.view_matrix, 16)
            camera_position[:] = memory.read_float(camera_ptr + Offsets.Camera.position, 3)

            if scraped_info.get('bomb_impact_point') is not None:
                bomb_impact_point[:] = scraped_info['bomb_impact_point']

            render_engine.begin_drawing()
            render_engine.clear_background(render_engine.BLANK)

            if window_info.get('window_active') is None:
                render_engine.end_drawing()
                continue
            if (prev_window_info is None or prev_window_info['window_active'] is False) and window_info[
                'window_active'] is True:
                show_window(overlay_handle, True, active=False)
            if (prev_window_info is None or prev_window_info['window_active'] is True) and window_info[
                'window_active'] is False:
                show_window(overlay_handle, False)
            if (prev_window_info is None or prev_window_info['active_class'] != 'GLFW30') and window_info[
                'active_class'] == 'GLFW30':
                remove_topmost_flag(overlay_handle)
                activate_window_by_class('DagorWClass')
                restore_topmost_flag(overlay_handle)

            prev_window_info = window_info.copy()
            if window_info.get('window_active') is not True:
                render_engine.end_drawing()
                continue

            label = "CykaWare v" + Settings.Product.version + " | " + Settings.Product.name + " "
            label_width = render_engine.measure_text(label.encode(), Settings.Render.text_height)
            draw_text(int(screen_w - label_width), 2, label, text_center=False)

            if local_unit is None:
                render_engine.end_drawing()
                continue

            local_position[:] = local_unit['position']
            local_velocity[:] = local_unit['velocity']
            local_acceleration[:] = local_unit['acceleration']

            weapon_position[:] = local_unit['position'] + scraped_info['weapon']['weapon_position']
            weapon_position = rotate_points(weapon_position, local_unit['rotation'], local_position)

            # ingame_crosshair_position[:] = scraped_info['weapon'].get('weapon_position_alt')
            ingame_crosshair_position[:] = local_unit['position']
            ingame_crosshair_position[0] += 1000
            ingame_crosshair_position = rotate_points(ingame_crosshair_position, local_unit['rotation'],
                                                      local_unit['position'])


            weapon_to_crosshair_angle = get_direct_angle(weapon_position, ingame_crosshair_position)
            camera_to_crosshair_angle = get_direct_angle(camera_position, ingame_crosshair_position)

            # Show debug list
            if Settings.Render.is_debug:
                draw_debug_list(scraped_info)

            # Show counters
            counters_string = 'Alive: '
            for vehicle_class_name in scraped_info['count_by_class']:
                vehicle_count = scraped_info['count_by_class'][vehicle_class_name]
                counters_string += str(vehicle_count) + ' - ' + vehicle_class_name.replace('exp_', '').replace('bb_',
                                                                                                               '').replace(
                    'zeros', 'enjoyer') + 's | '
            counters_string = counters_string.strip(' | ')
            counters_string = counters_string.replace('zeros', 'humans')
            counters_string += ' '
            draw_text(4, 2, counters_string, text_center=False)

            # In game ballistics
            selected_unit = scraped_info['weapon'].get('selected_unit')

            # if scraped_info['weapon'].get('weapon_position_alt') is not None:
            #     ingame_crosshair_position[:] = scraped_info['weapon'].get('weapon_position_alt')
            w2s_ingame_crosshair = w2s(ingame_crosshair_position, view_matrix, screen_w, screen_h)
            if 5 < w2s_ingame_crosshair[0] < screen_w - 5:
                render_engine.draw_circle(w2s_ingame_crosshair[0], w2s_ingame_crosshair[1], 3, Settings.Colors.truly_visible)

            # Show enemies
            bomb_is_hit = False
            for enemy_ptr in enemies:
                enemy = enemies[enemy_ptr]
                fix_offset = 0

                # enemy_speed = np.linalg.norm(enemy['velocity'], axis=0)
                enemy_ballistics = ballistics_info.get(enemy_ptr)
                enemy_color = get_box_color(enemy['visibility_state'])

                if bomb_is_hit is False:
                    if dist(enemy['position'], bomb_impact_point) < 15:
                        bomb_is_hit = True

                if enemy_ballistics is None and local_unit['type'] == 0:
                    enemy_color = Settings.Colors.known

                if enemy['visibility_state'] == 'truly_visible' and enemy['invul_state'] == 1:
                    enemy_color = Settings.Colors.known

                # if enemy['team'] == local_unit['team']:
                #     enemy_color = Settings.Colors.teammate

                if enemy['is_moving']:
                    enemy['position'] += enemy['velocity'] * (time() - enemy['read_time'])

                # Show 3d box
                enemy_w2s = w2s(enemy['position'], view_matrix, screen_w, screen_h)
                if 5 < enemy_w2s[0] < screen_w - 5 and 5 < enemy_w2s[1] < screen_h - 5:
                    if enemy['type'] != 0:
                        draw_3d_box_smart(enemy['position'], enemy['rotation'], enemy['bb_min'], enemy['bb_max'],
                                      view_matrix, camera_position, screen_w, screen_h,
                                      enemy_color)

                # Show text labels
                if 0 < enemy_w2s[0] < screen_w:
                    main_info = enemy['vehicle_name'] + " | " + str(round(enemy['dist'] / 1000, 2)) + "km"
                    main_info_width = int(render_engine.measure_text(main_info.encode(), 10))
                    text_position[:] = enemy_w2s
                    if text_position[0] <= 5:
                        text_position[0] += main_info_width * 0.5

                    if text_position[0] >= screen_w - 5:
                        text_position[0] -= main_info_width * 0.5

                    if text_position[1] >= screen_h - 5:
                        text_position[1] -= 10

                    if enemy['reload_time'] != 0:
                        render_engine.draw_rectangle(int(text_position[0] - (main_info_width*0.5) - 3),
                                                     int(text_position[1] + 17),
                                                     int(main_info_width + 8),
                                                     14, Settings.Colors.backdrop)
                        to_ready_width = (enemy['reload_time']/enemy['max_reload_time']) * main_info_width
                        render_engine.draw_rectangle(int(text_position[0] - (main_info_width * 0.5) - 3) + 4,
                                                     int(text_position[1] + 18),
                                                     int(to_ready_width),
                                                     10, render_engine.RED)
                        render_engine.draw_text('Reloading'.encode(), int(text_position[0] - (reload_text_width * 0.5)), int(text_position[1]) + 18, 10, render_engine.WHITE)

                    enemy_draw_infos = [main_info]
                    if enemy['invul_state'] == 1:
                        enemy_draw_infos.append("Invulnerable")
                    if Settings.Render.is_debug:
                        enemy_draw_infos.append("Flags : " + hex(enemy['flags']))
                        enemy_draw_infos.append("Unit ptr: " + hex(enemy['unit_ptr']))

                    draw_enemy_info(text_position, enemy_draw_infos, enemy_color)

                # Show ballistics marker
                if selected_unit == enemy_ptr and Settings.Ballistics.is_arcade is True:
                    if scraped_info['weapon'].get('ready') is True:
                        if scraped_info['weapon'].get('in_game') is not None and (
                                scraped_info['weapon'].get('in_game')[0] != 0 or
                                scraped_info['weapon'].get('in_game')[2] != 0):
                            predicted_position[:] = scraped_info['weapon'].get('in_game')
                            final_predicted_position[:] = predicted_position
                            if enemy['type'] == 0:
                                final_predicted_position[0] += 0.2
                                final_predicted_position[1] += 0.2
                            else:
                                final_predicted_position[1] += 1.35
                            final_predicted_position = rotate_points(final_predicted_position, enemy['rotation'],
                                                                     predicted_position)

                            impact_point[:] = enemy['position']
                            if enemy['type'] == 0:
                                impact_point[1] += 0.2
                            else:
                                impact_point[1] += 1.35
                            impact_point = rotate_points(impact_point, enemy['rotation'], enemy['position'])
                            w2s_impact_point = w2s(impact_point, view_matrix, screen_w, screen_h)

                            w2s_in_game_marker = w2s(final_predicted_position, view_matrix, screen_w,
                                                     screen_h)
                            if 0 < w2s_in_game_marker[0] < screen_w and 0 < w2s_impact_point[0] < screen_w:
                                render_engine.draw_line(w2s_impact_point[0], w2s_impact_point[1],
                                                        w2s_in_game_marker[0], w2s_in_game_marker[1],
                                                        enemy_color)
                                render_engine.draw_circle(w2s_in_game_marker[0], w2s_in_game_marker[1], 4,
                                                          enemy_color)
                else:
                    if enemy_ballistics is not None:
                        # Get linear predicted fly time and ballistics angle values
                        current_dist = dist(weapon_position, enemy['position'])
                        fly_time = (enemy_ballistics['fly_time']/enemy_ballistics['calc_dist'])*current_dist
                        ballistic_angle = (enemy_ballistics['angle_offset']/enemy_ballistics['calc_dist'])*current_dist

                        # Get enemy position in time
                        predicted_position = enemy['position'] + (
                            enemy['velocity']) * fly_time + 0.5 * (enemy['acceleration']) * fly_time ** 2

                        # Add ballistics angle
                        direct_angle = get_direct_angle(weapon_position, predicted_position)
                        xz_dist = dist([weapon_position[0], 0, weapon_position[2]],
                                       [predicted_position[0], 0, predicted_position[2]])
                        height_offset = xz_dist * tan(direct_angle[1] + ballistic_angle)
                        height_offset = ((predicted_position[1] - weapon_position[1]) - height_offset)
                        predicted_position[1] -= height_offset

                        # Subtract local speed
                        if local_unit['type'] == 0:
                            direction_vector = predicted_position - weapon_position
                            direction_vector_magnitude = np.linalg.norm(direction_vector)
                            normalized_direction_vector = direction_vector / direction_vector_magnitude

                            original_velocity_magnitude = np.linalg.norm(local_unit['velocity'])
                            if original_velocity_magnitude != 0:
                                velocity_towards_target[:] = np.dot(local_unit['velocity'],
                                                                    normalized_direction_vector) * normalized_direction_vector
                            else:
                                velocity_towards_target[:] = [0, 0, 0]

                            predicted_position -= (local_unit['velocity'] - velocity_towards_target) * fly_time
                            predicted_position -= (local_unit['acceleration'] * 0.5) * fly_time
                        else:
                            predicted_position -= local_unit['velocity'] * fly_time

                        # Add target offset from enemy bottom
                        final_predicted_position[:] = predicted_position
                        if enemy['type'] == 0:
                            final_predicted_position[1] += 0.1
                        else:
                            final_predicted_position[1] += 1.35
                        final_predicted_position = rotate_points(final_predicted_position, enemy['rotation'],
                                                                 predicted_position)

                        # Fix air crosshair convergence
                        if local_unit['type'] == 0:
                            xz_dist_weapon = dist([weapon_position[0], 0, weapon_position[2]],
                                                  [final_predicted_position[0], 0,
                                                   final_predicted_position[2]])
                            weapon_height_offset = xz_dist_weapon * tan(weapon_to_crosshair_angle[1])
                            weapon_height_offset -= (final_predicted_position[1] - weapon_position[1])

                            xz_dist_camera = dist([camera_position[0], 0, camera_position[2]],
                                                  [final_predicted_position[0], 0,
                                                   final_predicted_position[2]])
                            camera_height_offset = xz_dist_camera * tan(camera_to_crosshair_angle[1])
                            camera_height_offset -= (final_predicted_position[1] - camera_position[1])

                            fix_offset = camera_height_offset - weapon_height_offset
                            predicted_position[:] = final_predicted_position
                            final_predicted_position[1] += fix_offset

                        # Draw ballistics marker and history points
                        w2s_predicted_position = w2s(final_predicted_position, view_matrix, screen_w, screen_h)
                        if 5 < w2s_predicted_position[0] < screen_w - 5:
                            render_engine.draw_circle_lines(w2s_predicted_position[0], w2s_predicted_position[1], 5,
                                                            enemy_color)
                            pred_step = 0.25
                            pred_steps = ceil(fly_time/pred_step)
                            if height_offset != 0 and pred_steps != 0:
                                h_step = height_offset/pred_steps
                                fix_step = 0 if fix_offset == 0 else fix_offset/pred_steps
                                for i in range(pred_steps):
                                    pred_time = pred_step * i
                                    pred_h = h_step * i
                                    pred_fix = fix_step * i
                                    if pred_time == round(pred_time):
                                        dot_size = 3
                                    else:
                                        dot_size = 2
                                    predicted_position[:] = enemy['position'] + enemy['velocity'] * pred_time + 0.5 * enemy[
                                        'acceleration'] * pred_time ** 2

                                    if local_unit['type'] == 0:
                                        predicted_position -= (local_unit['velocity'] - velocity_towards_target) * pred_time
                                        predicted_position -= (local_unit['acceleration'] * 0.5) * pred_time
                                    else:
                                        predicted_position -= local_unit['velocity'] * pred_time

                                    if enemy['type'] == 0:
                                        predicted_position[1] += 0.1
                                    else:
                                        predicted_position[1] += 1.35
                                    predicted_position[1] -= pred_h
                                    predicted_position[1] += pred_fix
                                    w2s_predicted_position = w2s(predicted_position, view_matrix, screen_w,
                                                                 screen_h)
                                    if 0 < w2s_predicted_position[0] < screen_w:
                                        render_engine.draw_circle(w2s_predicted_position[0],
                                                                  w2s_predicted_position[1], dot_size,
                                                                  enemy_color)

            # Air markers
            if local_unit['type'] == 0:
                # Show bomb marker
                if bomb_impact_point[0] != 0 and bomb_impact_point[2] != 0:
                    bomb_impact_size[:] = bomb_impact_point
                    bomb_impact_size[0] += 30
                    bomb_impact_size[2] += 30
                    w2s_bomb_impact_point = w2s(bomb_impact_point, view_matrix, screen_w, screen_h)
                    w2s_bomb_impact_size = w2s(bomb_impact_size, view_matrix, screen_w, screen_h)
                    if 5 < w2s_bomb_impact_point[0] < screen_w - 5 and 5 < w2s_bomb_impact_size[0] < screen_w - 5:
                        bomb_impact_point_color = render_engine.Color(255, 255, 255, 100)
                        if bomb_is_hit:
                            bomb_impact_point_color = render_engine.Color(69, 248, 130, 200)
                        render_engine.draw_circle(w2s_bomb_impact_point[0], w2s_bomb_impact_point[1], 3,
                                                  bomb_impact_point_color)
                        render_engine.draw_circle_lines(w2s_bomb_impact_point[0], w2s_bomb_impact_point[1],
                                                        int(abs(w2s_bomb_impact_size[1] - w2s_bomb_impact_point[1])),
                                                        bomb_impact_point_color)

            render_engine.end_drawing()

        except KeyboardInterrupt as e:
            shared_exit.set()
            exit()

        except Exception as e:
            if get_process('aces.exe') is False:
                shared_exit.set()
                exit()

            if Settings.Memory.is_debug:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

            render_engine.end_drawing()
            continue
