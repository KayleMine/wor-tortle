from pynput.mouse import Listener, Button


class MouseListener:
    def __init__(self):
        self.right_button_pressed = False

        def on_click(x, y, button, pressed):
            if button == Button.right:
                self.right_button_pressed = pressed

        self.listener = Listener(on_click=on_click)
        self.listener.start()

    def is_right_button_pressed(self):
        return self.right_button_pressed

    def stop(self):
        self.listener.stop()
        self.listener.join()
