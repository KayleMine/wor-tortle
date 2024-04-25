from math import pi, dist, cos, sin, hypot, sqrt, atan2
import numpy as np
import numba as nb
import sys
import os



def qv_mult(q, v):
  qv = np.cross(q[1:], v)
  qv = q[0] * v + qv + np.cross(q[1:], qv)
  return qv


def transform(v, q):
  v_rotated = qv_mult(q, v)
  return v_rotated


def from_axis_angle(axis, angle):
  axis = axis / np.linalg.norm(axis) # нормализация оси
  sin_angle = np.sin(angle / 2)
  cos_angle = np.cos(angle / 2)

  return np.array([cos_angle,
                   axis[0] * sin_angle,
                   axis[1] * sin_angle,
                   axis[2] * sin_angle])


def align_vector(vector, target_vector):
    if np.linalg.norm(vector) == 0 or np.linalg.norm(target_vector) == 0:
        return vector

    # Normalize the vectors
    vector = vector / np.linalg.norm(vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    # Calculate the cross product, which will be the axis of rotation
    rotation_axis = np.cross(vector, target_vector)

    # If the vectors are parallel or anti-parallel, no rotation is needed
    if np.allclose(rotation_axis, 0):
        return vector

    # Calculate the angle between the vectors
    angle = np.arccos(np.clip(np.dot(vector, target_vector), -1.0, 1.0))

    # Create a rotation matrix for the given angle and axis
    rotation_matrix = np.array([
        [np.cos(angle) + rotation_axis[0] ** 2 * (1 - np.cos(angle)),
         rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
         rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
        [rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
         np.cos(angle) + rotation_axis[1] ** 2 * (1 - np.cos(angle)),
         rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
        [rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
         rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
         np.cos(angle) + rotation_axis[2] ** 2 * (1 - np.cos(angle))]
    ])

    # Apply the rotation matrix to the vector
    aligned_vector = np.dot(vector, rotation_matrix)

    return aligned_vector



def rotate_vector(vector, angles):
    # angles should be an array [x_angle, y_angle]
    x_angle, y_angle = angles

    # Create rotation matrices
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, cos(x_angle), -sin(x_angle)],
        [0, sin(x_angle), cos(x_angle)]
    ])

    rotation_matrix_y = np.array([
        [cos(y_angle), 0, sin(y_angle)],
        [0, 1, 0],
        [-sin(y_angle), 0, cos(y_angle)]
    ])

    # Apply rotation matrices to the vector
    rotated_vector = np.dot(rotation_matrix_x, vector)
    rotated_vector = np.dot(rotation_matrix_y, rotated_vector)

    return rotated_vector



# @nb.njit
def get_direct_angle(shooter_position, target_position):
    direction = target_position - shooter_position
    direction = direction / np.linalg.norm(direction)
    x_angle = atan2(direction[2], direction[0]) * -1
    y_angle = atan2(-direction[1], sqrt(direction[0] ** 2 + direction[2] ** 2)) * -1
    if x_angle < -pi:
        x_angle += 2 * pi
    elif x_angle > pi:
        x_angle -= 2 * pi
    if y_angle < -pi:
        y_angle += 2 * pi
    elif y_angle > pi:
        y_angle -= 2 * pi
    return [x_angle, y_angle]

def get_angle_between_vectors(v1, v2):
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)

    x_angle = atan2(v2_normalized[2] - v1_normalized[2], v2_normalized[0] - v1_normalized[0])
    y_angle = atan2(v2_normalized[1] - v1_normalized[1], sqrt((v2_normalized[0] - v1_normalized[0]) ** 2 + (v2_normalized[2] - v1_normalized[2]) ** 2))

    return [x_angle, y_angle]


def rotate_vector_to_target(point_a, point_b, velocity_a):
    # Вычисляем вектор от точки A к точке B
    vector_ab = point_b - point_a

    # Вычисляем углы между вектором скорости и вектором AB
    angles_velocity = get_angle_between_vectors(velocity_a, [1, 0, 0])
    angles_ab = get_angle_between_vectors(vector_ab, [1, 0, 0])

    # Вычисляем разницу между углами
    angles_diff = angles_ab - angles_velocity

    # Поворачиваем вектор скорости на разницу углов
    rotated_velocity = rotate_vector(velocity_a, angles_diff)

    return rotated_velocity


# @nb.njit
def get_2d_angle(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    angle = atan2(dy, dx)

    return angle


# @nb.njit
def predict_position_acc(cur_pos, cur_vel, prev_vel, cur_time, prev_time, X):
        dt = cur_time - prev_time
        if dt > 0:
            acc = (cur_vel - prev_vel) / dt
            predicted_vel = cur_vel + acc * X
            predicted_pos = cur_pos + cur_vel * X + 0.5 * acc * X ** 2
            return predicted_pos
        return cur_pos + (cur_vel * X)


def predict_velocity(cur_vel, prev_vel, cur_time, prev_time, X):
    dt = cur_time - prev_time
    if dt > 0 and prev_time != 0:
        acc = (cur_vel - prev_vel) / dt
        # predicted_vel = cur_vel + acc * X
        predicted_vel = cur_vel + acc
        return predicted_vel
    return cur_vel


def get_acceleration(cur_vel, prev_vel, cur_time, prev_time):
    dt = cur_time - prev_time
    if dt > 0 and prev_time != 0 and np.linalg.norm(cur_vel, axis=0) > 0:
        acc = (cur_vel - prev_vel) / dt
        return acc
    return np.zeros(3)

def get_predicted_position(unit, time):
    enemy_dt = unit['read_time'] - unit['delayed_time']
    enemy_vel_diff = unit['velocity'] - unit['delayed_velocity']

    if enemy_dt != 0 and unit['delayed_time'] != 0:
        enemy_vel_diff_speed = np.linalg.norm(enemy_vel_diff)
        if enemy_vel_diff_speed != 0:
            enemy_acc = np.divide(enemy_vel_diff, enemy_dt, out=np.zeros_like(enemy_vel_diff), where=enemy_dt != 0)
        else:
            enemy_acc = np.array([0, 0, 0])
    else:
        enemy_acc = np.array([0, 0, 0])

    enemy_acc = np.nan_to_num(enemy_acc, nan=0.0)
    predicted_position = unit['position'] + (unit['velocity'] * time) + (0.5 * enemy_acc * time ** 2)
    return predicted_position



def calculate_acceleration_old(time_history, velocity_history):
    # Создание массивов времени и скорости
    time_array = np.array(time_history)
    velocity_array = np.array(velocity_history)

    # Аппроксимация полиномом второй степени методом наименьших квадратов
    try:
        if time_array[0] - time_array[len(time_array) - 1] == 0 and np.linalg.norm(velocity_array[0] - velocity_array[len(velocity_array) - 1]) != 0:
            return [0,0,0]
        coefficients = np.polyfit(time_array, velocity_array, 1)
        # Ускорение равно второму коэффициенту полинома
        acceleration = 2 * coefficients[0]

        return acceleration
    except:
        return [0,0,0]

def calculate_acceleration(time_history, velocity_history):
    # Преобразование списков в массивы NumPy
    time_array = np.array(time_history)
    velocity_array = np.array(velocity_history)
    if time_array[0] - time_array[len(time_array) - 1] == 0:
        return [0,0,0]

    # Нормализация времени
    time_mean = np.mean(time_array)
    time_std = np.std(time_array)
    time_normalized = (time_array - time_mean) / time_std

    # Аппроксимация полиномом второй степени методом наименьших квадратов
    coefficients = np.polyfit(time_normalized, velocity_array, 2)

    # Ускорение равно второму коэффициенту полинома
    acceleration = 2 * coefficients[1]

    # Денормализация ускорения
    acceleration /= time_std

    return acceleration

# @nb.njit
def predict_smart_position(prev_pos, prev_vel, prev_acc, delta_time, pos, vel, acc, prediction_time):
    # Расчет изменения скорости и положения
    # delta_vel = vel - prev_vel
    delta_pos = pos - prev_pos
    # return pos + vel * prediction_time + 0.5 * acc * (prediction_time ** 2)
    # Расчет скорости и положения в момент времени prediction_time
    # predicted_vel = vel + delta_vel * (prediction_time / delta_time)
    if delta_time > 0:
        # predicted_pos = pos + delta_pos * (prediction_time / delta_time) + 0.5 * acc * (prediction_time ** 2)
        # predicted_pos = pos + predicted_vel * (prediction_time / delta_time) + 0.5 * acc * (prediction_time ** 2)
        predicted_pos = pos + vel * prediction_time + 0.5 * (acc + prev_acc) / 2 * (prediction_time ** 2) + 1 / 6 * (
                    acc - prev_acc) / delta_time * (prediction_time ** 3)
        # predicted_pos = pos + vel * prediction_time + 0.5 * acc * (prediction_time ** 2)
    else:
        predicted_pos = pos + vel * prediction_time + 0.5 * acc * (prediction_time ** 2)

    return predicted_pos

@nb.njit
def w2s(pos, matrix, screen_w, screen_h):
    result = [0, 0]
    w = pos[0] * matrix[3] + pos[1] * matrix[7] + pos[2] * matrix[11] + matrix[15]
    x = pos[0] * matrix[0] + pos[1] * matrix[4] + pos[2] * matrix[8] + matrix[12]
    y = pos[0] * matrix[1] + pos[1] * matrix[5] + pos[2] * matrix[9] + matrix[13]

    if w < 0.1:
        w = 1.0 / -w
        is_back = True
    else:
        is_back = False
        w = 1.0 / w

    nx = x * w
    ny = y * w

    if is_back is True:
        result[0] = int((screen_w / 2 * nx) + (nx + screen_w / 2))
        result[1] = int(-(screen_h / 2 * ny) + (ny + screen_h / 2))
        if result[1] < screen_h * 0.4:
            result[1] = 0
        else:
            result[1] = screen_h
    else:
        result[0] = int((screen_w / 2 * nx) + (nx + screen_w / 2))
        result[1] = int(-(screen_h / 2 * ny) + (ny + screen_h / 2))

    if result[0] <= 0:
        result[0] = 5
    if result[1] <= 0:
        result[1] = 5
    if result[0] >= screen_w:
        result[0] = screen_w - 5
    if result[1] >= screen_h:
        result[1] = screen_h - 5
    return result


def py_w2s_batch(positions, matrix, screen_w, screen_h):
    result = []
    for position in positions:
        ready_w2s = w2s(position, matrix, screen_w, screen_h)
        result.append(ready_w2s)
        del ready_w2s
    return result


def w2s_batch(positions, matrix, screen_w, screen_h):
    positions = np.array(positions)
    screen_positions = np.full((len(positions), 2), False, dtype=int)
    w = positions[:, 0] * matrix[3] + positions[:, 1] * matrix[7] + positions[:, 2] * matrix[11] + matrix[15]
    mask = w > 0.1
    w_masked = w[mask]
    x_masked = positions[mask, 0] * matrix[0] + positions[mask, 1] * matrix[4] + positions[mask, 2] * matrix[8] + \
               matrix[12]
    y_masked = positions[mask, 0] * matrix[1] + positions[mask, 1] * matrix[5] + positions[mask, 2] * matrix[9] + \
               matrix[13]
    nx = x_masked * (1.0 / w_masked)
    ny = y_masked * (1.0 / w_masked)
    screen_positions[mask, 0] = ((screen_w / 2 * nx) + (nx + screen_w / 2)).astype(int)
    screen_positions[mask, 1] = (-(screen_h / 2 * ny) + (ny + screen_h / 2)).astype(int)
    screen_positions[:, 0] = np.clip(screen_positions[:, 0], 5, screen_w - 5)
    screen_positions[:, 1] = np.clip(screen_positions[:, 1], 5, screen_h - 5)
    screen_positions[~mask] = False
    return screen_positions


# @nb.njit
def rotate_points(points, rotation, position):
    rotation_matrix = np.array(rotation).reshape(3, 3)
    rel_points = np.array(points) - np.array(position)
    rotated_points = np.dot(rel_points, rotation_matrix) + np.array(position)
    del rel_points
    del rotation_matrix
    return rotated_points


# @nb.njit
def generate_cube(position, bb_mins, bb_maxs):
    # Generate cube
    bb_min = list(map(sum, zip(position, bb_mins)))
    bb_max = list(map(sum, zip(position, bb_maxs)))

    # Generate points along each axis
    x = [bb_max[0], bb_min[0]]
    y = [bb_max[1], bb_min[1]]
    z = [bb_max[2], bb_min[2]]

    # Create meshgrid
    cube_points = []
    for xi in x:
        for zi in z:
            for yi in y:
                cube_points.append([xi, yi, zi])

    del bb_min
    del bb_max
    del x
    del y
    del z
    result = list(map(list, cube_points))
    del cube_points
    return result


# @nb.njit
def get_cube_faces(points):
    return [
        [points[2], points[3], points[1], points[0]],
        [points[4], points[5], points[7], points[6]],
        [points[0], points[1], points[5], points[4]],
        [points[6], points[7], points[3], points[2]],
        [points[4], points[6], points[2], points[0]],
        [points[1], points[3], points[7], points[5]]
    ]


# @nb.njit
def get_cube_edges(points):
    return [
        (points[0], points[1]),
        (points[1], points[3]),
        (points[3], points[2]),
        (points[2], points[0]),

        (points[4], points[5]),
        (points[5], points[7]),
        (points[7], points[6]),
        (points[6], points[4]),

        (points[0], points[4]),
        (points[1], points[5]),
        (points[2], points[6]),
        (points[3], points[7])
    ]


# @nb.njit
def calculate_face_normal(face):
    v0 = np.array(face[0])
    v1 = np.array(face[1])
    v2 = np.array(face[2])
    edge1 = v1 - v0
    edge2 = v2 - v0
    del v0
    del v1
    del v2
    return np.cross(edge1, edge2)


# @nb.njit
def find_face_center(face):
    num_points = len(face)
    if num_points == 0:
        return None
    center = np.sum(face, axis=0)
    center = center / num_points
    del num_points
    return center


# @nb.njit
def get_3d_box_visible_faces(position, rotation, unit_min, unit_max, camera_position, matrix, screen_w, screen_h):
    cube_points = generate_cube(position, unit_min, unit_max)
    rotated_points = rotate_points(cube_points, rotation, position)
    cube_faces = get_cube_faces(rotated_points)
    # Prepare flat for batch
    flat_array = [x for sublist in cube_faces for x in sublist]
    w2s_flat = py_w2s_batch(flat_array, matrix, screen_w, screen_h)
    w2s_edges = [w2s_flat[i:i+4] for i in range(0, len(w2s_flat), 4)]

    # Get faces
    visible_faces = []
    for face_i, face in enumerate(cube_faces):
        v0 = np.array(face[0])
        v1 = np.array(face[1])
        v2 = np.array(face[2])
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = py_cross(edge1, edge2)
        face_center = py_mean(face)
        camera_to_center = sub_arrays(face_center, camera_position)
        # Add only visible edges
        if np.dot(normal, camera_to_center) > 0:
            visible_faces.append(w2s_edges[face_i])

    del cube_points
    del rotated_points
    del cube_faces
    del flat_array
    del w2s_flat
    del w2s_edges

    return visible_faces


# @nb.njit
def get_3d_box_edges(position, rotation, bb_min, bb_max, matrix, screen_w, screen_h):
    cube_points = generate_cube(position, bb_min, bb_max)
    rotated_points = rotate_points(cube_points, rotation, position)
    cube_edges = get_cube_edges(rotated_points)
    flat_array = list([x for sublist in cube_edges for x in sublist])
    w2s_flat = list(py_w2s_batch(flat_array, matrix, screen_w, screen_h))
    w2s_edges = list([w2s_flat[i:i + 2] for i in range(0, len(w2s_flat), 2)])
    del w2s_flat
    del flat_array
    del cube_edges
    del rotated_points
    del cube_points
    return w2s_edges

@nb.njit
def py_cross(left, right):
    return [((left[1] * right[2]) - (left[2] * right[1])), ((left[2] * right[0]) - (left[0] * right[2])), ((left[0] * right[1]) - (left[1] * right[0]))]


# @nb.njit
def py_mean(values):
    return tuple(t / len(values) for t in [sum(col) for col in zip(*values)])


@nb.njit
def sub_arrays(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
    return result


def py_meshgrid(x, y, z):
    X = []
    Y = []
    Z = []

    for i in range(len(y)):
        row_X = []
        row_Y = []
        row_Z = []

        for j in range(len(x)):
            row_X.append(x[j])
            row_Y.append(y[i])
            row_Z.append(z[0])

        X.append(row_X)
        Y.append(row_Y)
        Z.append(row_Z)

    return X, Y, Z


def py_flatten(array):
    return [x for row in array for x in row]


def transpose(array):
    return [[row[i] for row in array] for i in range(len(array[0]))]


def py_vstack(arrays):
    arr1, arr2 = arrays
    return arr1 + arr2


def from_axis_angle(axis, angle):
  axis = axis / np.linalg.norm(axis)
  sin_angle = np.sin(angle / 2)
  cos_angle = np.cos(angle / 2)

  return np.array([cos_angle,
                   axis[0] * sin_angle,
                   axis[1] * sin_angle,
                   axis[2] * sin_angle])

