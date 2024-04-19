from math import pi, dist, cos, sin, hypot, ceil, sqrt
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from functions.maths import get_direct_angle, get_2d_angle
from settings import Settings

g = -9.8
time_step = 1 / 96
# time_step = 1 / 50
target_deviation = 0.1


@nb.njit
def get_drag_constant(shoot_height):
    unk_1 = float(2.2871901e-19)
    unk_2 = float(5.8355603e-14)
    unk_3 = float(0.00000000353118)
    unk_4 = float(0.000095938703)
    # max_alt = float(18300.0)
    max_alt = float(18300.0)
    # alt_mult = float(1.225)
    alt_mult = float(1.000)
    clamped_alt = min(shoot_height, max_alt)
    return (alt_mult * ((max_alt / max(shoot_height, max_alt))
                        * ((((((
                                       unk_1 * clamped_alt) - unk_2) * clamped_alt) + unk_3) * clamped_alt) - unk_4)
                        * clamped_alt + 1.0))


def get_density(height):
    height_clamped = min(height, 20000)
    base_value = 20000 * 1.225
    polynomial = height_clamped * 2.2872e-19
    polynomial += -5.8356e-14
    polynomial *= height_clamped
    polynomial += 0.0000000035312
    polynomial *= height_clamped
    polynomial += -0.000095939
    polynomial *= height_clamped
    density = base_value + (polynomial * base_value)
    density /= max(20000, height)
    return density

@nb.njit
def get_ballistics_coefficient(drag_constant, shell_caliber, shell_length, shell_mass):
    return -1.0 * (
            drag_constant * pi * 0.5
            * pow(float(shell_caliber) * 0.5, 2.0)
            * float(shell_length)
    ) / float(shell_mass)


target_2d_position = np.zeros(2)
shooter_2d_position = np.zeros(2)
hit_2d_position = np.zeros(2)

def find_ballistics_angle(shooter_position, target_position, shell_velocity, ballistic_coefficient, prev_angle = None, min_angle = None, max_angle = None):
    if max_angle is None:
        max_angle = pi / 2

    if min_angle is None:
        min_angle = -pi / 2

    is_hit = False
    is_too_far = False
    best_try_angle = None
    best_try_fly_time = 999999
    best_try_deviation = 9999999

    angle = prev_angle
    if prev_angle is None:
        angle = (min_angle + max_angle) / 2

    xz_dist = dist([shooter_position[0], shooter_position[2]], [target_position[0], target_position[2]])
    target_2d_position[:] = [xz_dist, target_position[1]]
    shooter_2d_position[:] = [0, shooter_position[1]]
    direct_angle = get_2d_angle(shooter_2d_position, target_2d_position)
    find_try = 0
    while abs(max_angle - min_angle) > 0.0000001 and find_try < 20:
        find_try += 1
        trajectory_result = check_ballistics_trajectory(angle,
                                                        shell_velocity,
                                                        shooter_position,
                                                        target_position,
                                                        ballistic_coefficient)


        # Show result curve if in debug
        if Settings.Ballistics.is_debug:
            print(trajectory_result)
            xz_dist = dist([shooter_position[0], shooter_position[2]], [target_position[0], target_position[2]])
            target_2d_position[:] = [xz_dist, target_position[1]]
            plt.plot(trajectory_result['bullet_history']['x'], trajectory_result['bullet_history']['y'], 'ro')
            plt.plot(target_2d_position[0], target_2d_position[1], 'go')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Angle: ' + str(angle) + '; Deviation: ' + str(trajectory_result['deviation']) + '; FlyTime: ' + str(trajectory_result['fly_time']))
            plt.grid()
            plt.show()

        # Too far target
        if trajectory_result['is_too_far'] is True:
            is_too_far = True
            break

        # Found better angle
        if trajectory_result['deviation'] < best_try_deviation:
            best_try_deviation = trajectory_result['deviation']
            best_try_angle = angle
            best_try_fly_time = trajectory_result['fly_time']

        # HIT
        if trajectory_result['is_hit'] is True:
            is_hit = True
            break

        if trajectory_result['is_low'] is True:
            min_angle = angle
            if trajectory_result['is_end'] is True:
                hit_2d_position[:] = [trajectory_result['bullet_history']['x'][-1],
                                      trajectory_result['bullet_history']['y'][-1]]
                target_angle = get_2d_angle(shooter_2d_position, target_2d_position)
                bullet_angle = get_2d_angle(shooter_2d_position, hit_2d_position)
                angle -= (bullet_angle - target_angle)
                continue

        if trajectory_result['is_over'] is True:
            max_angle = angle
            if trajectory_result['is_end'] is True:
                hit_2d_position[:] = [trajectory_result['bullet_history']['x'][-1],
                                      trajectory_result['bullet_history']['y'][-1]]
                target_angle = get_2d_angle(shooter_2d_position, target_2d_position)
                bullet_angle = get_2d_angle(shooter_2d_position, hit_2d_position)
                angle -= (bullet_angle - target_angle)
                continue

        angle = (min_angle + max_angle) / 2

    return {
        'direct_angle': direct_angle,
        'best_try_angle': best_try_angle,
        'best_try_deviation': best_try_deviation,
        'fly_time': best_try_fly_time,
        'is_hit': is_hit,
        'is_too_far': is_too_far
    }


# @nb.njit
def check_ballistics_trajectory(angle, shell_velocity, shooter_position, target_position, ballistic_coefficient):
    is_end = False
    is_too_far = False
    is_hit = False
    is_over = False
    is_low = False

    x1, y1 = shooter_position[0], shooter_position[2]
    x2, y2 = target_position[0], target_position[2]
    dx = x1 - x2
    dy = y1 - y2
    xz_dist = (dx ** 2 + dy ** 2) ** 0.5

    target_2d_position = [xz_dist, target_position[1]]

    # Prepare initial bullet velocity
    vx = float(shell_velocity) * cos(angle)
    vy = float(shell_velocity) * sin(angle)

    # Prepare initial bullet position and time
    bullet_fly_time = 0
    x = 0
    y = shooter_position[1]
    xs = []
    ys = []
    current_deviation = 30000

    if target_2d_position[0] / shell_velocity > 20:
        is_too_far = True
        xs.append(1)
        ys.append(1)

    # Simulate bullet loop
    frames_count = 0
    while is_too_far is False:
        bullet_fly_time += time_step  # Increase bullet fly time
        frames_count += 1

        if frames_count > 10000 or bullet_fly_time > 40:
            is_too_far = True
            break

        # Prepare velocity
        # Get velocity mult
        delta_speed = (ballistic_coefficient * time_step) * hypot(vx, vy)
        velocity_mult = (delta_speed / (1.0 - delta_speed)) + 1.0
        vx = vx * velocity_mult
        vy = (g * time_step) + (vy * velocity_mult)

        # Affect to bullet and save position
        x += (vx * time_step)
        y += (vy * time_step)

        x1, y1 = x, y
        x2, y2 = target_2d_position[0], target_2d_position[1]
        dx = x1 - x2
        dy = y1 - y2
        current_deviation = (dx ** 2 + dy ** 2) ** 0.5

        xs.append(x)
        ys.append(y)

        next_delta_speed = (ballistic_coefficient * time_step) * hypot(vx, vy)
        next_velocity_mult = (next_delta_speed / (1.0 - next_delta_speed)) + 1.0
        next_vx = vx * next_velocity_mult
        next_vy = (g * time_step) + (vy * next_velocity_mult)
        next_x = x + (next_vx * time_step)
        next_y = y + (next_vy * time_step)

        # change_x = next_x - x
        # change_y = next_y - y
        # predict_frames_x = (target_2d_position[0] - next_x) / change_x
        # predict_frames_y = (target_2d_position[1] - next_y) / change_y
        # if predict_frames_y < predict_frames_x and abs(next_vy) > next_vx and next_vy < 0 and next_x + (change_x * predict_frames_y) < target_2d_position[0]:
        #     is_too_far = True
        #     break

        # Is end
        if next_x >= target_2d_position[0] or current_deviation < target_deviation:
            is_end = True
            dx = target_2d_position[0] - x
            time_to_target = dx / next_vx
            bullet_fly_time += time_to_target
            x = x + next_vx * time_to_target
            y = y + next_vy * time_to_target
            xs.append(x)
            ys.append(y)

            x1, y1 = x, y
            x2, y2 = target_2d_position[0], target_2d_position[1]
            dx = x1 - x2
            dy = y1 - y2
            current_deviation = (dx ** 2 + dy ** 2) ** 0.5

            # Is hit
            if abs(target_2d_position[1] - y) <= target_deviation:
                is_hit = True
                break

            if y > target_2d_position[1]:
                # Is over
                is_over = True
                break
            else:
                # Is low
                is_low = True
                break

        # Is too far
        if vy <= 0 and vx <= 0 and (x < target_2d_position[0] or y < target_2d_position[1]):
            is_too_far = True
            break

    if Settings.Ballistics.is_debug:
        print('Final angle frames: ',frames_count)


    result = {
        'fly_time': bullet_fly_time,
        'angle': angle,
        'is_end': is_end,
        'is_too_far': is_too_far,
        'is_hit': is_hit,
        'is_over': is_over,
        'is_low': is_low,
        'bullet_position': {'x': x, 'y': y},
        'target_position': {'x': target_2d_position[0], 'y': target_2d_position[1]},
        'deviation': current_deviation,
    }

    if Settings.Ballistics.is_debug:
        result['bullet_history'] = {'x': xs, 'y': ys}
    else:
        result['bullet_history'] = {'x': [xs[-1]], 'y': [ys[-1]]}

    return result


