
import raypyc as render_engine


class Settings:
    class Product:
        name = 'Arcade free'
        version = '8.6'
        server = 'https://warchill.xyz'

    class Memory:
        is_debug = False
        process_name = 'aces.exe'
        memory_method = 'usermode'

    class Scraper:
        is_debug = False
        is_death_match = False
        target_fps = 60

    class Render:
        is_debug = False
        target_fps = 60
        text_height = 10
        enable_msaa_x4 = True
        enable_vsync = True

    class MultiProcessing:
        is_debug = False

    class Ballistics:
        is_debug = False
        is_arcade = True
        target_fps = 60
        max_fly_time = 4

    class Aim:
        target_fps = 144

    class Window:
        is_debug = False
        target_fps = 10

    class Colors:
        shadow = render_engine.Color(0, 0, 0, 120)  # Black
        invisible = render_engine.Color(255, 65, 68, 230)  # Red box
        known = render_engine.Color(253, 249, 0, 120)  # Yellow box
        visible = render_engine.Color(64, 197, 43, 40)  # Green box
        truly_visible = render_engine.Color(142, 250, 138, 150)  # Green box
        scouted = render_engine.Color(179, 46, 219, 200)  # Purple box
        teammate = render_engine.Color(63, 106, 248, 200)  # Purple box
        cached = render_engine.Color(255, 255, 255, 100)
        backdrop = render_engine.Color(11, 15, 25, 170)


