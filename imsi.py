import pyglet

window = pyglet.window.Window()

cursor = window.get_system_mouse_cursor(window.CURSOR_HELP)
window.set_mouse_cursor(cursor)

pyglet.app.run()