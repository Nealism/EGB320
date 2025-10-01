# -*- coding:utf-8 -*-

from __future__ import print_function
import sys
import os
from pynput import keyboard
from pynput.keyboard import Key
sys.path.append("../")

import time

from DFRobot_RaspberryPi_DC_Motor import THIS_BOARD_TYPE, DFRobot_DC_Motor_IIC as Board

from collection import level_1, level_2, level_3, centre, open_grab, close_grab, initServos

board = Board(1, 0x10)

# Global duty cycle variable
duty = 75
duty_step = 2  # how much to increase/decrease with +/-

def board_detect():
    l = board.detect()
    print("Board list conform:")
    print(l)

def print_board_status():
    if board.last_operate_status == board.STA_OK:
        print("board status: everything ok")
    elif board.last_operate_status == board.STA_ERR:
        print("board status: unexpected error")
    elif board.last_operate_status == board.STA_ERR_DEVICE_NOT_DETECTED:
        print("board status: device not detected")
    elif board.last_operate_status == board.STA_ERR_PARAMETER:
        print("board status: parameter error, last operate no effective")
    elif board.last_operate_status == board.STA_ERR_SOFT_VERSION:
        print("board status: unsupported board firmware version")

def on_key_release(key):
    board.motor_stop(board.ALL)

def on_key_pressed(key):
    global duty

    if key == Key.right:
        print(f"Right key pressed at duty {duty}%")
        board.motor_movement([board.M1], board.CW, duty)
        board.motor_movement([board.M2], board.CW, duty)

    elif key == Key.left:
        print(f"Left key pressed at duty {duty}%")
        board.motor_movement([board.M1], board.CCW, duty)
        board.motor_movement([board.M2], board.CCW, duty)

    elif key == Key.up:
        print(f"Up key pressed at duty {duty}%")
        board.motor_movement([board.M1], board.CCW, duty)
        board.motor_movement([board.M2], board.CW, duty)

    elif key == Key.down:
        print(f"Down key pressed at duty {duty}%")
        board.motor_movement([board.M1], board.CW, duty)
        board.motor_movement([board.M2], board.CCW, duty)

    elif key == Key.esc:
        print("ESC pressed: stopping motors for 3s")
        board.motor_stop(board.ALL)
        time.sleep(3)

    elif key == Key.delete:
        print("Delete pressed: exiting in 3s")
        time.sleep(3)
        listener.stop()

    elif hasattr(key, 'char') and key.char=='o':
        open_grab()
    
    elif hasattr(key, 'char') and key.char == 'c':
        close_grab()

    elif hasattr(key, 'char') and key.char == 'b':
        level_1()
        
    elif hasattr(key, 'char') and key.char == 'n':
        level_2()

    elif hasattr(key, 'char') and key.char == 'm':
        level_3()
    
    

    # Adjust duty with + and -
    elif hasattr(key, 'char') and key.char == '+':
        duty = min(100, duty + duty_step)
        print(f"Duty increased to {duty}%")

    elif hasattr(key, 'char') and key.char == '-':
        duty = max(0, duty - duty_step)
        print(f"Duty decreased to {duty}%")

if __name__ == "__main__":
    print("Starting...")

    initServos()
    centre()

    with keyboard.Listener(on_press=on_key_pressed, on_release=on_key_release) as listener:
        listener.join()

    board_detect()

    while board.begin() != board.STA_OK:
        print_board_status()
        print("board begin failed")
        time.sleep(2)

    print("board begin success")
    board.set_encoder_enable(board.ALL)
    board.set_encoder_reduction_ratio(board.ALL, 43)
    board.set_moter_pwm_frequency(1000)