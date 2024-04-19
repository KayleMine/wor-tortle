import raypyc as render_engine
import numpy as np
from copy import copy

from functions.maths import get_3d_box_visible_faces

from settings import Settings


def draw_enemy_info(enemy_w2s, infos, color):
    info_id = 0
    for info in infos:
        draw_text(int(enemy_w2s[0]), int(enemy_w2s[1]) + 5 + (14 * info_id), str(info), color)
        info_id += 1
    del info_id
    return True

def draw_text(x, y, info_text, color=Settings.Colors.visible, with_bg=True, text_center=True):
    color = copy(color)
    color.a = 200
    left_offset = 0
    info_text_width = render_engine.measure_text(str(info_text).encode(), Settings.Render.text_height)
    if text_center:
        left_offset = int(info_text_width / 2)
    if with_bg:
        render_engine.draw_rectangle_rounded(render_engine.Rectangle(x - left_offset - 5, y - 2, info_text_width + 10, Settings.Render.text_height + 4), 2.0, 10, Settings.Colors.backdrop)
        # render_engine.draw_rectangle(x - left_offset - 4,
        #                              y - 2,
        #                              info_text_width + 8,
        #                              Settings.Render.text_height + 4, Settings.Colors.backdrop)
    render_engine.draw_text(str(info_text).encode(), x - left_offset,
                            y, Settings.Render.text_height, color)


def draw_3d_box_smart(position, rotation, bb_min, bb_max, matrix, camera_position, screen_w, screen_h, box_color=render_engine.Color(69, 248, 130, 255)):
    # Get visible faces
    w2s_faces = get_3d_box_visible_faces(position, rotation, bb_min, bb_max, camera_position, matrix, screen_w, screen_h)
    if w2s_faces is False:
        return False

    # Draw visible faces
    for face in w2s_faces:
        if (face[0][0] == 0 or
                face[1][0] == 0 or
                face[2][0] == 0 or
                face[3][0] == 0 or
                face[0][1] == 0 or
                face[1][1] == 0 or
                face[2][1] == 0 or
                face[2][1] == 0 or
                face[3][1] == 0):
            continue

        # Prepare line tick
        line_tick = 2.0

        # Draw primary lines
        render_engine.draw_line_ex(render_engine.Vector2(face[0][0], face[0][1]),
                             render_engine.Vector2(face[1][0], face[1][1]),
                             line_tick, box_color)
        render_engine.draw_line_ex(render_engine.Vector2(face[1][0], face[1][1]),
                             render_engine.Vector2(face[2][0], face[2][1]),
                             line_tick, box_color)
        render_engine.draw_line_ex(render_engine.Vector2(face[2][0], face[2][1]),
                             render_engine.Vector2(face[3][0], face[3][1]),
                             line_tick, box_color)
        render_engine.draw_line_ex(render_engine.Vector2(face[3][0], face[3][1]),
                             render_engine.Vector2(face[0][0], face[0][1]),
                             line_tick, box_color)
    del w2s_faces
    return True


def draw_debug_list(scraped_info):
    debug_list_x = 200
    render_engine.draw_fps(5, 5)
    debug_list = [
        "Total units: " + str(len(scraped_info['all_units_ptrs'])),
        "Perform units: " + str(len(scraped_info['perform_units'])),
        "Render units: " + str(len(scraped_info['enemy_units'])),
        "Skip units: " + str(len(scraped_info['skip_unit_ptrs'])),
        "Entity count units: " + str(scraped_info['entity_list_count']),
        "Weapon velocity: " + str(scraped_info['weapon'].get('velocity')),
    ]
    if scraped_info.get('local_unit') is not None:
        debug_list.append("Local unit: " + hex(scraped_info['local_unit']['unit_ptr']))
        debug_list.append("Local entity: " + hex(scraped_info['local_unit']['entity_ptr']))
        debug_list.append("Local velocity: " + str(scraped_info['local_unit']['velocity']))
        debug_list.append("Local speed: " + str(round(np.linalg.norm(scraped_info['local_unit']['velocity']) * 0.001 * 3600,2)) + " km/h")

    param_id = 0
    for param in debug_list:
        draw_text(debug_list_x, 100 + (14 * param_id), str(param))
        param_id += 1


def get_box_color(visibility_state):
    if visibility_state == 'invisible':
        return Settings.Colors.invisible
    if visibility_state == 'truly_visible':
        return Settings.Colors.truly_visible
    if visibility_state == 'visible':
        return Settings.Colors.truly_visible
    if visibility_state == 'cached':
        return Settings.Colors.cached
    if visibility_state == 'scouted':
        return Settings.Colors.scouted
    if visibility_state == 'known':
        return Settings.Colors.known

