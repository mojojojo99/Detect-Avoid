#!/usr/bin/python3

# adapted from https://github.com/recantha/EduKit3-RC-Keyboard/blob/master/rc_keyboard.py

import sys, termios, tty, os, time
import fcntl

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

button_delay = 0.2

fd = sys.stdin.fileno()
fl = fcntl.fcntl(fd, fcntl.F_GETFL)
fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

while True:
    char = getch()

    if (char == "p"):
        print("Stop!")
        exit(0)

    if (not char):
        print("No char")
    elif (char == "a"):
        print("Left pressed")
        time.sleep(button_delay)

    elif (char == "d"):
        print("Right pressed")
        time.sleep(button_delay)

    elif (char == "w"):
        print("Up pressed")
        time.sleep(button_delay)

    elif (char == "s"):
        print("Down pressed")
        time.sleep(button_delay)

    elif (char == "1"):
