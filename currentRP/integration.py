from __future__ import print_function
import sys
import os
sys.path.append("../")

from time import sleep, monotonic, time
import types
import traceback
from enum import IntEnum
import math
import numpy as np

from DFRobot_RaspberryPi_DC_Motor import THIS_BOARD_TYPE, DFRobot_DC_Motor_IIC as Board

'''
subsystem integration modules
'''
from cameraModules.processing_module import process_frame, Obstacle
from cameraModules.camera_module import Camera
import cv2

import lgpio
from collection import level_1, level_2, level_3, centre, open_grab, close_grab, initServos


if THIS_BOARD_TYPE:
  board = Board(1, 0x10)    # RaspberryPi select bus 1, set address to 0x10
else:
  board = Board(7, 0x10)    # RockPi select bus 7, set address to 0x10

def board_detect():
  l = board.detecte()
  print("Board list conform:")
  print(l)

''' print last operate status, users can use this variable to determine the result of a function call. '''
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
    print("board status: unsupport board framware version")


class RobotMode(IntEnum):
   SEARCH_MARKER = 1
   DRIVE_TO_BAY = 2
   ARRIVE_BAY = 3
   DROP_ITEM = 4
   DEBUG_STOP = 5
   DEBUG_TEST = 6

# physical measurements in mm
WHEEL_DIAMETER_MM = 30
TRACK_WIDTH_MM = 170

LEFT_MOTOR_ID =  board.M1
RIGHT_MOTOR_ID = board.M2
LEFT_FORWARD_ORIENT = board.CW
RIGHT_FORWARD_ORIENT = board.CCW

def drive_distance(board, distance_m, speed_pct, wheel_diameter_m=WHEEL_DIAMETER_MM, gear_ratio=50,
                   slow_down_window_m=0.10, min_speed_pct=15.0, poll_dt=0.01,
                   watchdog_s=1.5):
    """
    Drive straight for distance_m (meters) at speed_pct (%) and stop.
    Positive distance_m => CW, negative => CCW.
    Uses get_encoder_speed() [RPM] and integrates to distance.
    wheel_diameter_m: physical wheel diameter in meters.
    gear_ratio: set to your gearbox ratio so the HAT reports *wheel* RPM (e.g., 50).

    Returns the measured distance traveled (meters).
    """

    board.set_encoder_enable([board.M1, board.M2])
    board.set_encoder_reduction_ratio([board.M1, board.M2], int(gear_ratio))

    target = abs(float(distance_m))*0.75
    speed_pct = float(speed_pct)
    speed_pct_M2 = 0.85*speed_pct # slight motor imbalance
    circ = math.pi * float(wheel_diameter_m/1000)  # wheel circumference

    # Safety clamps
    speed_pct = max(0.0, min(100.0, speed_pct))
    min_speed_pct = max(0.0, min(min_speed_pct, speed_pct))

    board.motor_movement([board.M1], board.CCW, speed_pct)
    board.motor_movement([board.M2], board.CW, speed_pct_M2)

    sL = sR = 0.0
    s_last_change_t = monotonic()
    t_prev = s_last_change_t

    try:
        while True:
            sleep(poll_dt)
            t_now = monotonic()
            dt = t_now - t_prev
            t_prev = t_now

            rpmL, rpmR = board.get_encoder_speed([board.M1, board.M2])

            # Convert RPM -> revolutions advanced over dt.
            # Use absolute in case encoder sign differs from motor orientation wiring.
            revL = abs(rpmL) * dt / 60.0
            revR = abs(rpmR) * dt / 60.0

            # Distance per wheel
            sL += revL * circ
            sR += revR * circ

            # Chassis distance = average of wheel distances
            s = 0.5 * (sL + sR)

            # Watchdog: detect if weÃ¢â‚¬â„¢ve effectively stopped moving
            if (revL + revR) > 0:
                s_last_change_t = t_now
            elif (t_now - s_last_change_t) > watchdog_s:
                # Not moving for too long -> break to avoid hanging
                break

            remaining = target - s
            if remaining <= 0:
                break

            # 4) Simple taper near the goal to reduce overshoot
            if remaining < slow_down_window_m:
                scaled = speed_pct * (remaining / slow_down_window_m)
                cmd_speed = max(min_speed_pct, scaled)
                board.motor_movement([board.M1], board.CCW, speed_pct)
                board.motor_movement([board.M2], board.CW, speed_pct_M2)

    finally:
        board.motor_stop(board.ALL)

    return 0.5 * (sL + sR)

def turnAngle_seconds(angle, duty=100):
   # 3 seconds at duty cycle of 100 is approximately 250 degrees
   # therefore 1.083 seconds at duty cycle 100 for 90 degrees
   # 0.012 seconds per degree

    tConst = 0.012*2

    timeDelta = abs(angle) * tConst

    if angle > 0:
        direction = 'right'
    else:
       direction = 'left'

    if direction == 'right':
        board.motor_movement([board.M1], board.CW, duty)
        board.motor_movement([board.M2], board.CW, duty)
    elif direction == 'left':
       board.motor_movement([board.M1], board.CCW, duty)
       board.motor_movement([board.M2], board.CCW, duty)

    sleep(timeDelta)

    board.motor_stop(board.ALL)


def turnAngle_encoder(board,
                      angle_deg,
                      speed_pct,
                      wheel_diameter_mm=WHEEL_DIAMETER_MM,
                      wheelbase_mm=170,
                      gear_ratio=50,
                      slow_down_window_deg=20.0,
                      min_speed_pct=15.0,
                      poll_dt=0.01,
                      watchdog_s=1.5):

    # 1) Make HAT report wheel RPM
    board.set_encoder_enable([board.M1, board.M2])
    board.set_encoder_reduction_ratio([board.M1, board.M2], int(gear_ratio))

    # 2) Geometry
    circ = (1/2)*math.pi * (wheel_diameter_mm / 1000.0)           # m
    b = wheelbase_mm / 1000.0                               # m
    theta_rad = abs(angle_deg) * math.pi / 180.0

    # You integrate s = 0.5*(|sL|+|sR|), so target must be (b*theta)/2
    target = 0.5 * b * theta_rad                            # m
    slow_down_window_m = 0.5 * b * (slow_down_window_deg * math.pi / 180.0)

    # 3) Directions (match your convention: +angle => right)
    if angle_deg >= 0:
        oL = board.CW
        oR = board.CW
    else:
        oL = board.CCW
        oR = board.CCW

    speed_pct = float(max(0.0, min(100.0, speed_pct)))
    min_speed_pct = float(max(0.0, min(min_speed_pct, speed_pct)))

    board.motor_movement([board.M1], oL, speed_pct)
    board.motor_movement([board.M2], oR, speed_pct)

    sL = sR = 0.0
    t_prev = monotonic()
    s_last_change_t = t_prev

    try:
        while True:
            sleep(poll_dt)
            t_now = monotonic()
            dt = t_now - t_prev
            t_prev = t_now                    # <-- CRITICAL: update t_prev!

            rpmL, rpmR = board.get_encoder_speed([board.M1, board.M2])

            # Distance this step (use absolute values)
            sL += abs(rpmL) * dt / 60.0 * circ
            sR += abs(rpmR) * dt / 60.0 * circ

            s_avg = 0.5 * (sL + sR)

            # Watchdog (not moving?)
            if (abs(rpmL) + abs(rpmR)) > 0:
                s_last_change_t = t_now
            elif (t_now - s_last_change_t) > watchdog_s:
                break

            remaining = target - s_avg
            if remaining <= 0:
                break

            # Taper near target
            if remaining < slow_down_window_m:
                scaled = speed_pct * (remaining / slow_down_window_m)
                cmd_speed = max(min_speed_pct, scaled)
                board.motor_movement([board.M1], oL, cmd_speed)   # <-- use cmd_speed
                board.motor_movement([board.M2], oR, cmd_speed)

    finally:
        board.motor_stop(board.ALL)

    # Report measured angle (degrees) from encoders
    theta_meas_rad = (2.0 * s_avg) / b
    return math.degrees(theta_meas_rad)
    


# shelf helpers
def calcShelfMarkersRB(obstacles):
    shelfMarkers = []
    for obstacle in obstacles:
        if obstacle.type_name == "Row Marker":
            shelfMarkers.append(obstacle)
    #shelfMarkers = Obstacle.filter_by_(obstacles,1)
    shelfMarkerRB = {0: None, 1: None, 2: None}
    if shelfMarkers:
        print('shelfMarkers in function is', shelfMarkers)
        for shelf in shelfMarkers:
            shelfMarkerRB[shelf.station_id-1] = (shelf.distance, shelf.bearing[0])
    return shelfMarkerRB


'''
TARGETS
'''
TARGET_SHELF_INDEX = 1
if TARGET_SHELF_INDEX > -1 and TARGET_SHELF_INDEX < 2:
   TARGET_ROW_INDEX = 0
elif TARGET_SHELF_INDEX > 1 and TARGET_SHELF_INDEX < 4:
   TARGET_ROW_INDEX = 1
elif TARGET_SHELF_INDEX > 3 and TARGET_SHELF_INDEX < 6:
   TARGET_ROW_INDEX = 2

TARGET_BAY = 2
TARGET_BAY_HEIGHT = 2
BEARING_LIMIT = 5


def calcBayDistance():
   reverseBayID = abs(TARGET_BAY-4)
   distance = (reverseBayID-1)*26 + 13
   return distance/100


def steerToBearing(b, limit=BEARING_LIMIT):
    if b > limit:
        board.motor_movement([board.M1],board.CW, 50)
        board.motor_movement([board.M2], board.CW, 50)
        return False
    elif b < -limit:
        board.motor_movement([board.M1],board.CCW, 50)
        board.motor_movement([board.M2], board.CCW, 50)
    else:
        board.motor_stop(board.ALL)
        return True


'''
CAMERA OVERLAY CHECKERS
'''
def draw_overlay(frame, obstacles, trajectory, plot_options):
    """Draw detection overlays on the frame"""
    if plot_options.get('plot_contours', False):
        for obstacle in obstacles:
            x, y = obstacle.centre[0], obstacle.centre[1]
            
            # Draw shape based on available information and settings
            if plot_options.get('plot_exact_contours', True) and obstacle.contour is not None:
                # Draw exact contour if available
                cv2.drawContours(frame, [obstacle.contour], -1, obstacle.colour_bgr, 2)
            elif plot_options.get('plot_approximated_shapes', False):
                # Fall back to approximated shapes if exact contours not available or not wanted
                if obstacle.shape == "circle":
                    radius = int(np.sqrt(obstacle.area / np.pi))
                    cv2.circle(frame, obstacle.centre, radius, obstacle.colour_bgr, 2)
                else:
                    aspect_ratio = 1.2
                    height = int(np.sqrt(obstacle.area / aspect_ratio))
                    width = int(height * aspect_ratio)
                    x1 = x - width//2
                    y1 = y - height//2
                    cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), obstacle.colour_bgr, 2)
            
            # Draw center point if enabled
            if plot_options.get('plot_centers', True):
                cv2.circle(frame, obstacle.centre, 3, obstacle.colour_bgr, -1)
            
            # Draw text labels if enabled
            if plot_options.get('plot_labels', True):
                # Draw object type and ID (if applicable)
                label_parts = []
                label_parts.append(obstacle.type_name)
                if obstacle.station_id is not None:
                    label_parts.append(f"({obstacle.station_id})")
                if obstacle.distance is not None:
                    label_parts.append(f"{obstacle.distance:.2f}m")
                
                # First line: Type, ID, and distance
                main_label = " ".join(label_parts)
                cv2.putText(frame, main_label,
                          (x + 10, y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          obstacle.colour_bgr, 2)
            
                        
                if obstacle.bearing is not None:
                    # Bearing on third line
                    bear_text = f"Bearing: ({obstacle.bearing[0]:.1f}, {obstacle.bearing[1]:.1f})"
                    cv2.putText(frame, bear_text,
                        (x + 10, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        obstacle.colour_bgr, 2)

    if plot_options['plot_trajectory'] and trajectory:
        # Draw trajectory points
        for point in trajectory:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)

    if plot_options['plot_centre_reference']:
        height, width = frame.shape[:2]
        centrex, centrey = width // 2, height // 2
        cv2.line(frame, (centrex, 0), (centrex, height), (255, 255, 255), 1)
        cv2.line(frame, (0, centrey), (width, centrey), (255, 255, 255), 1)

    return frame


'''
GRABBER MACROS
'''

def setGrabber():
    centre()
    open_grab()
    if TARGET_BAY_HEIGHT == 0:
        level_1()
    elif TARGET_BAY_HEIGHT == 1:
        level_2()
    elif TARGET_BAY_HEIGHT == 2:
        level_3()


if __name__ == "__main__":
    print("am i even etnering")
    try:
        print("start of try")
        board_detect()


        while board.begin() != board.STA_OK:    # Board begin and check board status
            print_board_status()
            print("board begin faild")
            sleep(2)
        print("board begin success")

        board.set_encoder_enable(board.ALL)                 # Set selected DC motor encoder enable
        # board.set_encoder_disable(board.ALL)              # Set selected DC motor encoder disable
        board.set_encoder_reduction_ratio(board.ALL, 43)    # Set selected DC motor encoder reduction ratio, test motor reduction ratio is 43.8

        board.set_moter_pwm_frequency(1000)   # Set DC motor pwm frequency to 1000HZ

        # initialise camera processing
        camera = Camera()

        # Target frequency in Hz
        target_freq = 20.0
        period = 1.0 / target_freq

        initServos()


        robotMode = RobotMode.SEARCH_MARKER

        #robotMode = RobotMode.DEBUG_TEST

        plot_options = {
            'print_data': False,
            'plot_contours': True,          # Master switch for all shape drawing
            'plot_exact_contours': True,    # Draw exact contours when available
            'plot_approximated_shapes': False,  # Fall back to approximated shapes
            'plot_centers': True,           # Draw center points
            'plot_labels': True,            # Draw text labels
            'plot_trajectory': False,        # Draw trajectory
            'plot_centre_reference': False   # Draw center reference lines
        }

        print("HELP")

        while True:
            print('robotMode', robotMode)
            loop_start = time()
            frame = camera.capture_frame()
            obstacles, trajectory = process_frame(frame)

            frame_with_overlay = draw_overlay(frame.copy(), obstacles, trajectory, plot_options)            
            
            # Calculate and display FPS
            current_fps = 1.0 / (time() - loop_start)
            #fps_smooth = fps_smooth * (1 - fps_alpha) + current_fps * fps_alpha
            cv2.putText(frame_with_overlay, f'FPS: {current_fps:.1f}', 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display results
            cv2.imshow('Vision System', frame_with_overlay)

            shelfMarkersRB = calcShelfMarkersRB(obstacles)
            print('shelfMarkerRB is', shelfMarkersRB)

            #print("robot mode is ", robotMode)

            #sleep(2)


            if robotMode == RobotMode.DEBUG_TEST:
                #turnAngle_encoder(board, angle_deg=90, speed_pct=100)
                #turnAngle_seconds(-90)
                #drive_distance(board, distance_m=0.1, speed_pct=50)
                open_grab()
                sleep(2)
                close_grab()
                robotMode = RobotMode.DEBUG_STOP

            if robotMode == RobotMode.DEBUG_STOP:
                board.motor_stop(board.ALL)
                # lgpio.gpio_write(h, SERVO_LEFT, 0)
                # lgpio.gpio_write(h, SERVO_RIGHT, 0)
                # lgpio.gpio_write(h, SERVO_GRAB, 0)
                # lgpio.gpiochip_close(h)
            
            if robotMode == RobotMode.SEARCH_MARKER:
                close_grab()
                #print('shelfMarker:', shelfMarkersRB)
                if shelfMarkersRB[TARGET_ROW_INDEX]:
                    rowmB = shelfMarkersRB[TARGET_ROW_INDEX][1]
                    rowmD = shelfMarkersRB[TARGET_ROW_INDEX][0]
                    aligned = steerToBearing(rowmB)
                    if aligned:
                        board.motor_stop(board.ALL)
                        robotMode = RobotMode.DRIVE_TO_BAY
                else:
                  print("Not Found")
                  board.motor_movement([board.M1], board.CW, 50)
                  board.motor_movement([board.M2], board.CW, 50)

            if robotMode == RobotMode.DRIVE_TO_BAY:
               if shelfMarkersRB[TARGET_ROW_INDEX]:
                  rowmB = shelfMarkersRB[TARGET_ROW_INDEX][1]
                  rowmD = shelfMarkersRB[TARGET_ROW_INDEX][0]
                  if rowmD > calcBayDistance():
                     print("Driving to bay: "+str(rowmD))
                     drive_distance(board, distance_m=(rowmD-calcBayDistance()), speed_pct=75)
                     robotMode = RobotMode.ARRIVE_BAY
                  else:
                     print("Arrived at bay")
                     board.motor_stop(board.ALL)
                     robotMode = RobotMode.DEBUG_STOP

            if robotMode == RobotMode.ARRIVE_BAY:
               camera.close()
               cv2.destroyAllWindows()
               directionM = TARGET_SHELF_INDEX % 2
               if directionM == 0:
                  direction = 'left'
               elif directionM == 1:
                  direction = 'right'
               if direction == 'left':
                  turnAngle_encoder(board, angle_deg=-90, speed_pct=100)
                  setGrabber()
                  sleep(0.5)
                  drive_distance(board, distance_m=0.1, speed_pct=50)
                  sleep(0.5)
                  open_grab()
                  robotMode = RobotMode.DEBUG_STOP
               elif direction == 'right':
                  turnAngle_encoder(board, angle_deg=90, speed_pct=100)
                  setGrabber()
                  sleep(0.5)
                  drive_distance(board, distance_m=0.1, speed_pct=50)
                  sleep(0.5)
                  open_grab()
                  robotMode = RobotMode.DEBUG_STOP

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                plot_options['print_data'] = not plot_options['print_data']
            elif key == ord('l'):
                plot_options['plot_labels'] = not plot_options['plot_labels']
            elif key == ord('t'):
                plot_options['plot_trajectory'] = not plot_options['plot_trajectory']
            elif key == ord('c'):
                plot_options['plot_contours'] = not plot_options['plot_contours']
            elif key == ord('x'):
                plot_options['plot_exact_contours'] = not plot_options['plot_exact_contours']
            elif key == ord('a'):
                plot_options['plot_approximated_shapes'] = not plot_options['plot_approximated_shapes']
            elif key == ord('n'):
                plot_options['plot_centers'] = not plot_options['plot_centers']
            elif key == ord('r'):
                plot_options['plot_centre_reference'] = not plot_options['plot_centre_reference']

    except KeyboardInterrupt:
        print("\nShutting down...")
        board.motor_stop(board.ALL)
    except Exception as e:
       board.motor_stop(board.ALL)
       print(f"\nAn error occurred: {e}")
       traceback.print_exc()
    finally:
        board.motor_stop(board.ALL)
        # lgpio.gpio_write(h, SERVO_LEFT, 0)
        # lgpio.gpio_write(h, SERVO_RIGHT, 0)
        # lgpio.gpio_write(h, SERVO_GRAB, 0)
        # lgpio.gpiochip_close(h)
        camera.close()
        cv2.destroyAllWindows()

