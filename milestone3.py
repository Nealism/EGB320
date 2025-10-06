from mobility.direction import steerToBearing, calcBayDistance, drive_distance, turnAngle_encoder, rowIndex, turnSearch
from enum import IntEnum
from DFRobot_RaspberryPi_DC_Motor import THIS_BOARD_TYPE, DFRobot_DC_Motor_IIC as Board
import traceback
from time import sleep, monotonic, time

''' camera modules '''
from camera.processing_module import process_frame, Obstacle
from camera.camera_module import Camera
import cv2
from camera.RB import calcShelfMarkersRB, draw_overlay, calcPickingMarkersRB, calcPackingStationRB

''' collection modules '''
from collection import level_1, level_2, level_3, centre, open_grab, close_grab, initServos



# CONSTANTS (in m)
#TODO: MEASURE ALL OF THESE VALUES
STATION_ARRIVE_DIST = 0.08
STATION_DEPART_DIST = 0.5
RAMP_ARRIVE_DIST = 0.085
BEARING_LIMIT = 1
STATION_ROW_DIST = 1.1
SHELF_GAP = 0.55
STATION_DIST = 0.4

class RobotMode(IntEnum):
    SEARCH_STATION = 0
    CHECK_ROW = 1
    TO_RAMP = 2
    ASCEND_RAMP = 3
    COLLECT_ITEM = 4
    DESCEND_RAMP = 5
    SEARCH_SHELF = 6
    TO_SHELF = 7
    DROP_ITEM = 8
    EXIT_ROW = 9
    FIND_ORIGIN = 10
    DEBUG_STOP = 11

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




def bayTurn(TARGET_BAY):
    TARGET_SHELF_INDEX = TARGET_BAY[0]
    if TARGET_SHELF_INDEX % 2 == 0:
        direction = 'left'
    else:
        direction = 'right'

    return direction

# SIMULATED ORDER
orders = [(3, (1,2), 1), (2, (3,0), 2), (1, (5,3), 0) ]
ORDER_INDEX = 0

def processOrder():
    currentOrder = orders[ORDER_INDEX]
    TARGET_STATION_INDEX = currentOrder[0]
    TARGET_BAY = currentOrder[1]
    TARGET_BAY_HEIGHT = currentOrder[2]
    return TARGET_STATION_INDEX, TARGET_BAY, TARGET_BAY_HEIGHT


if __name__ == "__main__":
    try:
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

        robotMode = RobotMode.SEARCH_STATION

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
            pickingStationRB = calcPickingMarkersRB(obstacles)
            psRB = calcPackingStationRB(obstacles)
            print('shelfMarkerRB is', shelfMarkersRB)

            TARGET_STATION_INDEX, TARGET_BAY, TARGET_BAY_HEIGHT = processOrder()


            if robotMode == RobotMode.DEBUG_TEST:
                open_grab()
                sleep(2)
                close_grab()
                robotMode = RobotMode.DEBUG_STOP

            if robotMode == RobotMode.DEBUG_STOP:
                board.motor_stop(board.ALL)
            
            elif robotMode == RobotMode.SEARCH_STATION:
                psRBd = psRB[0]
                psRBb = psRB[1]
                if psRB:
                    if psRBd < RAMP_ARRIVE_DIST:
                        aligned = steerToBearing(psRBb)
                        if aligned:
                            board.motor_stop(board.ALL)
                            robotMode = RobotMode.CHECK_ROW
                    else:
                       drive_distance(board, psRBd - RAMP_ARRIVE_DIST, 100)
                else:
                   turnSearch(50, board)

            elif robotMode == RobotMode.CHECK_ROW:
                if shelfMarkersRB[1] != None:
                    row2B = shelfMarkersRB[1][1]
                    aligned = steerToBearing(row2B)
                    if aligned:
                        if shelfMarkersRB[1][0] > STATION_ROW_DIST:
                            drive_distance(board, shelfMarkersRB[1][0] - STATION_ROW_DIST, 100)
                        else:
                           board.motor_stop(board.ALL)
                           turnAngle_encoder(board, 90, 50)
                           robotMode = RobotMode.TO_RAMP
            

            elif robotMode == RobotMode.TO_RAMP:
                pickingBayDist = TARGET_STATION_INDEX * STATION_DIST
                drive_distance(board, pickingBayDist, 100)
                turnAngle_encoder(board, 90, 50)
                robotMode = RobotMode.ASCEND_RAMP

            elif robotMode == RobotMode.ASCEND_RAMP:
                pickingStation = pickingStationRB[TARGET_STATION_INDEX]
                if pickingStation:
                    pickingStationBearing = pickingStation[1]
                    pickingStationDistance = pickingStation[0]
                    if pickingStationDistance > STATION_ARRIVE_DIST:
                        aligned = steerToBearing(pickingStationBearing)
                        if aligned: 
                            drive_distance(board, pickingStationDistance - STATION_ARRIVE_DIST, 100)
                    else:
                       board.motor_stop(board.ALL)
                       robotMode = RobotMode.COLLECT_ITEM
                else:
                   turnSearch(50, board)

            
            elif robotMode == RobotMode.COLLECT_ITEM:
               board.motor_stop(board.ALL)
               #TODO: CHECK IF ROBOT CAN SEE OBJECT, CALC DIST FROM SAID OBJECT
               close_grab()
               robotMode = RobotMode.DESCEND_RAMP

            
            elif robotMode == RobotMode.DESCEND_RAMP:
                pickingStation = pickingStationRB[TARGET_STATION_INDEX]
                if pickingStation:
                    pickingStationBearing = pickingStation[1]
                    pickingStationDistance = pickingStation[0]
                    if pickingStationDistance < STATION_DEPART_DIST:
                        drive_distance(board, STATION_ARRIVE_DIST - pickingStationDistance, 100)
                    else:
                       board.motor_stop(board.ALL)
                       robotMode = RobotMode.SEARCH_SHELF
                else:
                   turnSearch(50, board)

            elif robotMode == RobotMode.SEARCH_SHELF:
                TARGET_ROW_INDEX = rowIndex(TARGET_BAY)
                if TARGET_STATION_INDEX == 3:
                    turnAngle_encoder(board, 90, 75)
                    if TARGET_ROW_INDEX == 1:
                        turnAngle_encoder(board, 90, 75)
                        robotMode = RobotMode.TO_SHELF
                    elif TARGET_ROW_INDEX == 2:
                        drive_distance(board, SHELF_GAP, 100)
                        turnAngle_encoder(board, 90, 75)
                        robotMode = RobotMode.TO_SHELF
                    elif TARGET_ROW_INDEX == 3:
                        drive_distance(board, SHELF_GAP*2, 100)
                        robotMode = RobotMode.TO_SHELF
                
                elif TARGET_STATION_INDEX == 2:
                    if TARGET_ROW_INDEX == 1:
                        turnAngle_encoder(board, -90, 75)
                        drive_distance(board, SHELF_GAP/2, 100)
                        turnAngle_encoder(board, -90, 75)
                        robotMode = RobotMode.TO_SHELF
                    elif TARGET_ROW_INDEX == 2:
                        turnAngle_encoder(board, 90, 75)
                        drive_distance(board, 2*SHELF_GAP/3, 100)
                        turnAngle_encoder(board, 90, 75)
                        robotMode = RobotMode.TO_SHELF
                    elif TARGET_ROW_INDEX == 3:
                        turnAngle_encoder(board, 90, 75)
                        drive_distance(board, SHELF_GAP*1.5, 100)
                        turnAngle_encoder(board, 90, 75)
                        robotMode = RobotMode.TO_SHELF

                elif TARGET_STATION_INDEX == 1:
                    if TARGET_ROW_INDEX == 1:
                        turnAngle_encoder(board, -90, 75)
                        drive_distance(board, (2*SHELF_GAP)/3, 100)
                        turnAngle_encoder(board, -90, 75)
                        robotMode = RobotMode.TO_SHELF
                    elif TARGET_ROW_INDEX == 2:
                        turnAngle_encoder(board, 90, 75)
                        drive_distance(board, SHELF_GAP/3, 100)
                        turnAngle_encoder(board, 90, 75)
                        robotMode = RobotMode.TO_SHELF
                    elif TARGET_ROW_INDEX == 3:
                        turnAngle_encoder(board, 90, 75)
                        drive_distance(board, SHELF_GAP, 100)
                        turnAngle_encoder(board, 90, 75)
                        robotMode = RobotMode.TO_SHELF

            
            elif robotMode == RobotMode.TO_SHELF:
                TARGET_BAY_INDEX = TARGET_BAY[1]
                if shelfMarkersRB[TARGET_ROW_INDEX] != None:
                    shelfB = shelfMarkersRB[TARGET_ROW_INDEX][1]
                    distIndex = -1*TARGET_BAY_INDEX  + 3
                    aligned = steerToBearing(shelfB)
                    BAY_GAP = calcBayDistance()
                    if distIndex+1 == 1:
                        bayDistance = (1/4)*BAY_GAP
                    else:
                        bayDistance = (1/2)*BAY_GAP + distIndex * BAY_GAP
                    if aligned:
                        if shelfMarkersRB[TARGET_ROW_INDEX][0] > bayDistance:
                            drive_distance(board, shelfMarkersRB[TARGET_ROW_INDEX][0] - bayDistance, 100)
                        else:
                            board.motor_stop(board.ALL)
                            robotMode = RobotMode.DROP_ITEM
                else:
                    turnSearch(50, board)

            
            elif robotMode == RobotMode.DROP_ITEM:
                if shelfMarkersRB[TARGET_ROW_INDEX] != None:
                    shelfB = shelfMarkersRB[TARGET_ROW_INDEX][1]
                    aligned = steerToBearing(shelfB)
                    if aligned:
                        direction = bayTurn(TARGET_BAY)
                        if direction == 'right':
                            turnAngle_encoder(board, -90, 75)
                        elif direction == 'left':
                            turnAngle_encoder(board, 90, 75)
                        if TARGET_BAY_HEIGHT == 0:
                            level_1()
                        elif TARGET_BAY_HEIGHT == 1:
                            level_2()
                        elif TARGET_BAY_HEIGHT == 2:
                            level_3()
                        drive_distance(board, 0.1, 50)
                        board.motor_stop(board.ALL)
                        sleep(1)
                        open_grab()
                        sleep(1)
                        drive_distance(board, -0.1, 50)
                        direction = bayTurn(TARGET_BAY)
                        if direction == 'right': angle = 90
                        elif direction == 'left': angle = -90
                        turnAngle_encoder(board, angle, 75)
                        aligned = steerToBearing(shelfB)
                        if aligned:
                            robotMode = RobotMode.EXIT_ROW
            

            elif robotMode == RobotMode.EXIT_ROW:
                if shelfMarkersRB[TARGET_ROW_INDEX] != None:
                    shelfD = shelfMarkersRB[TARGET_ROW_INDEX][0]
                    if shelfD < 5*STATION_ROW_DIST/8:
                        sleep(0.1)
                        drive_distance(board, 5*STATION_ROW_DIST/8 - shelfD, 100)
                    else:
                        board.motor_stop(board.ALL)
                        robotMode = RobotMode.FIND_ORIGIN
                else:
                    turnSearch(50, board)

            
            elif robotMode == RobotMode.FIND_ORIGIN:
                turnAngle_encoder(board, -90, 50)
                drive_distance(board, (SHELF_GAP)*(-TARGET_ROW_INDEX+3), 100)
                if ORDER_INDEX == 2:
                    robotMode = RobotMode.DEBUG_STOP
                else:
                    ORDER_INDEX += 1
                    robotMode = RobotMode.SEARCH_STATION
                        




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