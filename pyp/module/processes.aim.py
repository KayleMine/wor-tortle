import sys
import os
from os import system, getpid
from time import sleep, time
from math import sqrt, tan, dist, pi
import numpy as np
import ctypes
import threading

from functions.maths import get_direct_angle, rotate_points, predict_smart_position, from_axis_angle, transform
from functions.processes import get_process

from classes.camera import Camera

from helpers.memory import Memory
from helpers.mouse_listener import MouseListener
from helpers.fps_manager import FpsManager

from offsets import Offsets
from settings import Settings


def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


user32 = ctypes.windll.user32
mouse_event = user32.mouse_event

def left_click(delay=0.2):
    mouse_event(0x0002, 0, 0, 0, 0)
    sleep(delay)
    mouse_event(0x0004, 0, 0, 0, 0)

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



def is_key_pressed(key_code):
    return ctypes.windll.user32.GetAsyncKeyState(key_code) & 0x8000 != 0


def is_mouse_button_pressed(button_code):
    return ctypes.windll.user32.GetAsyncKeyState(button_code) & 0x8000 != 0


def aim_func(scraper_shared, ballistics_shared, window_shared, shared_exit, UserSettings):

    mouse_listener = MouseListener()
    if Settings.MultiProcessing.is_debug:
        print("[>] Aim process pid: " + str(getpid()))

    while get_process(Settings.Memory.process_name) is False:
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

    sleep(1)
    if shared_exit.is_set():
        sys.exit()
    scraped_info = scraper_shared.copy()['value']
    while scraped_info.get('camera_ptr') is None or scraped_info.get('camera_ptr') == 0:
        try:
            if shared_exit.is_set():
                sys.exit()
            scraped_info = scraper_shared.copy()['value']
            sleep(0.5)
        except KeyboardInterrupt:
            shared_exit.set()
            sys.exit()
    camera_ptr = scraped_info.get('camera_ptr')

    sleep(5)
    memory = Memory(Settings.Memory.process_name, Settings.Memory.memory_method)
    if memory.ready is False:
        shared_exit.set()
        sys.exit()
    camera = Camera(camera_ptr, memory)
    best_target = False

    predicted_position = np.zeros(3)
    final_predicted_position = np.zeros(3)
    custom_crosshair_position = np.zeros(3)
    local_position = np.zeros(3)
    local_velocity = np.zeros(3)
    unit_velocity = np.zeros(3)
    unit_acceleration = np.zeros(3)
    velocity_towards_target = np.zeros(3)
    acceleration_towards_target = np.zeros(3)
    local_acceleration = np.zeros(3)
    camera_position = np.zeros(3)
    aim_angle = np.zeros(2)
    prev_distance = np.zeros(2)
    aim_ex_offset = np.zeros(2)
    weapon_to_target_angle = np.zeros(2)
    weapon_to_crosshair_angle = np.zeros(2)
    camera_to_crosshair_angle = np.zeros(2)
    prev_is_saved = False
    prev_time = time()
    current_game_time = 1.0
    prev_game_time = 1.0
    time_speed_up = False
    time_speed_down = False
    prev_time_speed_up = False
    prev_time_speed_down = False
    weapon_position = np.zeros(3)
    weapon_offset = np.zeros(2)
    ingame_crosshair_position = np.zeros(3)

    is_active = mouse_listener.is_right_button_pressed()
    prev_is_active = is_active
    fps_manager = FpsManager('aim', Settings.Aim.target_fps)
    while shared_exit.is_set() is False:
        try:
            fps_manager.delay()

            window_info = window_shared.copy()['value']
            if window_info.get('window_active') is not True:
                if get_process('aces.exe') is False:
                    shared_exit.set()
                    sys.exit()
                continue

            scraped_info = scraper_shared.copy()['value']
            ballistics_info = ballistics_shared.copy()['value']
            enemies = scraped_info['enemy_units']
            camera_info = camera.update()
            camera_position[:] = camera.position
            local_unit = scraped_info.get('local_unit')
            if local_unit is None:
                continue

            if scraped_info.get('game_mode') == 'testFlight':
                time_speed_up = is_key_pressed(0x6A)  # key *
                if time_speed_up and prev_time_speed_up is False and current_game_time >= 1.0:
                    current_game_time = 15.0
                if (time_speed_up is False and prev_time_speed_up is True) or (time_speed_up is True and prev_time_speed_up is False and prev_game_time < 1.0):
                    current_game_time = 1.0

                time_speed_down = is_key_pressed(0x6F)  # key /
                if prev_time_speed_down is False and time_speed_down:
                    current_game_time -= 0.2

                if current_game_time < 0.2:
                    current_game_time = 0.2

                if prev_game_time != current_game_time:
                    memory.write_float(scraped_info['game_ptr'] + Offsets.Game.time, current_game_time)

                prev_game_time = current_game_time
                prev_time_speed_up = time_speed_up
                prev_time_speed_down = time_speed_down

            local_position[:] = local_unit['position']
            local_acceleration[:] = local_unit['acceleration']
            local_velocity[:] = local_unit['velocity']


            weapon_position[:] = local_unit['position'] + scraped_info['weapon']['weapon_position']
            weapon_position = rotate_points(weapon_position, local_unit['rotation'], local_position)

            ingame_crosshair_position[:] = local_unit['position']
            ingame_crosshair_position[0] += 1000
            ingame_crosshair_position = rotate_points(ingame_crosshair_position, local_unit['rotation'],
                                                      local_unit['position'])

            weapon_to_crosshair_angle = get_direct_angle(weapon_position, ingame_crosshair_position)
            camera_to_crosshair_angle = get_direct_angle(camera_position, ingame_crosshair_position)

            if Settings.Ballistics.is_arcade:
                selected_unit = scraped_info['weapon'].get('selected_unit')
            else:
                selected_unit = 0


            is_active = is_key_pressed(UserSettings.Aim.key) or is_mouse_button_pressed(UserSettings.Aim.key)
            if is_active != prev_is_active:
                prev_is_active = is_active
                best_target = False
                prev_angle_diff = None
                if is_active:
                    best_target = False
                    min_dist = float('inf')
                    prediction_factor = 1.0
                    prev_is_saved = False
                    prev_time = time()
                    aim_ex_offset[:] = [0, 0]
                    weapon_offset[:] = [0, 0]
                    if Settings.Ballistics.is_arcade is True:
                        if selected_unit != 0:
                            best_target = selected_unit
                    else:
                        for enemy_ptr in enemies:
                            enemy = enemies.get(enemy_ptr)
                            enemy_ballistics = ballistics_info.get(enemy_ptr)
                            if enemy is None:
                                continue

                            if enemy_ballistics is not None:
                                current_dist = dist(weapon_position, enemy['position'])
                                fly_time = (enemy_ballistics['fly_time'] / enemy_ballistics['calc_dist']) * current_dist
                                ballistic_angle = (enemy_ballistics['angle_offset'] / enemy_ballistics[
                                    'calc_dist']) * current_dist
                                predicted_position = enemy['position'] + (
                                    enemy['velocity']) * fly_time + 0.5 * (enemy['acceleration']) * fly_time ** 2

                                direct_angle = get_direct_angle(weapon_position, predicted_position)

                                xz_dist = dist([weapon_position[0], 0, weapon_position[2]],
                                               [predicted_position[0], 0, predicted_position[2]])
                                height_offset = xz_dist * tan(direct_angle[1] + ballistic_angle)
                                height_offset = ((predicted_position[1] - weapon_position[1]) - height_offset)
                                predicted_position[1] -= height_offset

                                if local_unit['type'] == 0:
                                    predicted_position -= (local_unit['velocity'] * 0.5) * fly_time
                                else:
                                    predicted_position -= (local_unit['velocity']) * fly_time
                                direct_angle = get_direct_angle(camera_info['position'], predicted_position)
                            else:
                                direct_angle = get_direct_angle(camera_info['position'], enemy['position'])

                            aim_dist = np.linalg.norm(direct_angle - camera_info['current_view'])
                            aim_dist = aim_dist * 360
                            if aim_dist < min_dist:
                                min_dist = aim_dist
                                best_target = enemy_ptr

            # Active aiming
            if is_active and best_target is not False:
                enemy = enemies.get(best_target)
                enemy_ballistics = ballistics_info.get(best_target)

                if (enemy_ballistics is None and Settings.Ballistics.is_arcade is False) or enemy is None:
                    best_target = False
                    continue

                if Settings.Ballistics.is_arcade is True:
                    predicted_position[:] = scraped_info['weapon'].get('in_game')
                else:
                    if enemy['is_moving']:
                        enemy['position'] += enemy['velocity'] * (time() - enemy['read_time'])

                    # Get linear predicted fly time and ballistics angle values
                    current_dist = dist(weapon_position, enemy['position'])
                    fly_time = (enemy_ballistics['fly_time'] / enemy_ballistics['calc_dist']) * current_dist
                    ballistic_angle = (enemy_ballistics['angle_offset'] / enemy_ballistics[
                        'calc_dist']) * current_dist

                    # Get enemy position in time
                    predicted_position[:] = predict_smart_position(enemy['delayed_position'],
                                                                   enemy['delayed_velocity'],
                                                                   enemy['delayed_acceleration'],
                                                                   (enemy['read_time'] - enemy['delayed_time']),
                                                                   enemy['position'],
                                                                   enemy['velocity'],
                                                                   enemy['acceleration'],
                                                                   fly_time)
                    # unit_velocity[:] = enemy['velocity']
                    # unit_acceleration[:] = enemy['acceleration']
                    # delta_time = enemy["read_time"] - enemy["delayed_time"]
                    # if 0 < delta_time:
                    #     if np.linalg.norm(enemy['velocity']) != 0 and np.linalg.norm(enemy['delayed_velocity']) != 0:
                    #         axis_velocity_cross = np.cross(enemy['velocity'], enemy['delayed_velocity']) / delta_time
                    #         if np.linalg.norm(axis_velocity_cross) != 0:
                    #             rotation_velocity_axis = axis_velocity_cross / np.linalg.norm(axis_velocity_cross)
                    #             rotation_velocity_angle = angle_between(enemy['velocity'],
                    #                                                     enemy['delayed_velocity']) * -1
                    #             rotation_velocity_angle *= fly_time
                    #             if rotation_velocity_angle != 0:
                    #                 rotation_quat_velocity = from_axis_angle(
                    #                     enemy['velocity'] + rotation_velocity_axis,
                    #                     rotation_velocity_angle)
                    #                 unit_velocity[:] = transform(enemy['velocity'], rotation_quat_velocity)
                    #
                    #     if np.linalg.norm(enemy['acceleration']) != 0 and np.linalg.norm(
                    #             enemy['delayed_acceleration']) != 0 and np.linalg.norm(
                    #         enemy['acceleration'] - enemy['delayed_acceleration']) != 0:
                    #         axis_acceleration_cross = np.cross(enemy['acceleration'],
                    #                                            enemy['delayed_acceleration']) / delta_time
                    #         if np.linalg.norm(axis_acceleration_cross) != 0:
                    #             rotation_acceleration_axis = axis_acceleration_cross / np.linalg.norm(
                    #                 axis_acceleration_cross)
                    #             rotation_acceleration_angle = angle_between(enemy['acceleration'],
                    #                                                         enemy['delayed_acceleration']) * -1
                    #             rotation_velocity_angle *= fly_time
                    #             if rotation_acceleration_angle != 0:
                    #                 rotation_quat_acceleration = from_axis_angle(
                    #                     enemy['acceleration'] + rotation_acceleration_axis,
                    #                     rotation_acceleration_angle)
                    #                 unit_acceleration[:] = transform(enemy['acceleration'],
                    #                                                  rotation_quat_acceleration)
                    #
                    #     predicted_position = enemy['position'] + (unit_velocity) * fly_time + 0.5 * (
                    #         unit_acceleration) * fly_time ** 2

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
                            # acceleration_towards_target[:] = local_unit['acceleration'] * normalized_direction_vector
                            acceleration_towards_target = np.dot(local_unit['acceleration'],
                                                                 normalized_direction_vector) * normalized_direction_vector

                        else:
                            velocity_towards_target[:] = [0, 0, 0]

                        # predicted_position -= (local_unit['velocity']) * fly_time
                        predicted_position -= (local_unit['velocity'] - velocity_towards_target) * fly_time
                        # predicted_position -= (local_unit['velocity']) * fly_time

                        # predicted_position -= local_unit['acceleration']
                        # predicted_position -= (local_unit['velocity'] - velocity_towards_target) * fly_time
                        # predicted_position -= (local_unit['acceleration'] * 0.5) * fly_time ** 2
                        # predicted_position -= (local_unit['acceleration'] * 0.5) * fly_time # good
                        # predicted_position -= (local_unit['acceleration'] * 0.25) * fly_time
                        # predicted_position -= (local_unit['acceleration'] * 0.5) * fly_time * 0.5 # good
                        # predicted_position -= (acceleration_towards_target * 0.5) * fly_time # very good
                        # predicted_position -= local_unit['acceleration'] + (acceleration_towards_target - local_unit['acceleration']) * fly_time # very smooth
                        # predicted_position -= local_unit['acceleration'] + (acceleration_towards_target - local_unit['acceleration']) * 0.5 * fly_time
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


                # Get target angle
                aim_angle[:] = get_direct_angle(camera_position, final_predicted_position)

                if scraped_info['weapon'].get('ready') is True and local_unit['type'] == 0:
                    weapon_to_target_angle[:] = get_direct_angle(weapon_position, final_predicted_position)

                    # current_meters_dist = dist(weapon_position, final_predicted_position)
                    # camera_angle_norm = weapon_to_crosshair_angle / np.linalg.norm(weapon_to_crosshair_angle)
                    # target_angle_norm = weapon_to_target_angle / np.linalg.norm(weapon_to_target_angle)
                    # angle_diff = np.arccos(np.clip(np.dot(camera_angle_norm, target_angle_norm), -1.0, 1.0))
                    # distance_deviation = current_meters_dist * np.sin(angle_diff)
                    # if distance_deviation < 0.1:
                    #     if is_mouse_button_pressed(0x01) is False:
                    #         threading.Thread(target=left_click, args=(0.1,)).start()

                    weapon_angle_dist = (weapon_to_target_angle - weapon_to_crosshair_angle)
                    current_time = time()
                    if prev_is_saved:
                        if current_time - prev_time > 0.05 / current_game_time:
                            if 0.000001 < np.linalg.norm(weapon_angle_dist) < 0.25:
                                if abs(prev_distance[0]) <= abs(weapon_angle_dist[0]):
                                    weapon_offset[0] += weapon_angle_dist[0] * 0.2
                                else:
                                    if 0 < weapon_offset[0]:
                                        weapon_offset[0] -= (abs(prev_distance[0]) - abs(weapon_angle_dist[0])) * 0.5
                                        weapon_offset[0] = max(0, weapon_offset[0])
                                    else:
                                        weapon_offset[0] += (abs(prev_distance[0]) - abs(weapon_angle_dist[0])) * 0.1
                                        weapon_offset[0] = min(0, weapon_offset[0])

                                if abs(prev_distance[1]) <= abs(weapon_angle_dist[1]):
                                    weapon_offset[1] += weapon_angle_dist[1] * 0.2
                                else:
                                    if 0 < weapon_offset[1]:
                                        weapon_offset[1] -= (abs(prev_distance[1]) - abs(weapon_angle_dist[1])) * 0.1
                                        weapon_offset[1] = max(0, weapon_offset[1])
                                    else:
                                        weapon_offset[1] += (abs(prev_distance[1]) - abs(weapon_angle_dist[1])) * 0.5
                                        weapon_offset[1] = min(0, weapon_offset[1])
                            else:
                                weapon_offset[:] = [0,0]
                            prev_time = current_time
                            prev_distance = weapon_angle_dist
                    else:
                        prev_time = current_time
                        prev_distance = weapon_angle_dist
                        prev_is_saved = True


                aim_angle += weapon_offset

                if aim_angle[0] < -pi:
                    aim_angle[0] += 2 * pi
                elif aim_angle[0] > pi:
                    aim_angle[0] -= 2 * pi
                if aim_angle[1] < -pi:
                    aim_angle[1] += 2 * pi
                elif aim_angle[1] > pi:
                    aim_angle[1] -= 2 * pi
                camera.set_current_view(aim_angle[0], aim_angle[1])

        except KeyboardInterrupt as e:
            shared_exit.set()
            sys.exit()
        except Exception as e:
            if get_process('aces.exe') is False:
                shared_exit.set()
                sys.exit()
            if Settings.Memory.is_debug:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

