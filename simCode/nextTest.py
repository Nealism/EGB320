from warehousebot_lib import *

import math, time, os, types
import numpy as np
import cv2
import sys
import types as _types
import traceback

DEBUG_TEST = False




# --- Windows/RPi-agnostic shim for camera_module (avoids picamera2 on Windows) ---
# Create a fake 'camera_module' so that 'from camera_module import Camera' in openCV/main.py works.
if 'camera_module' not in sys.modules:
    camera_module_stub = _types.ModuleType('camera_module')

    class Camera:
        """Stub Camera compatible with the expected API (no picamera2)."""
        def __init__(self, resolution=(1280, 720)):
            self.resolution = resolution

        def capture_frame(self):
            # Return a blank frame (not used in your flow, but keeps API intact)
            w, h = self.resolution
            return np.zeros((h, w, 3), dtype=np.uint8)

        def close(self):
            pass

    camera_module_stub.Camera = Camera
    sys.modules['camera_module'] = camera_module_stub
# -------------------------------------------------------------------------------

import openCV.processing_module as _pm
sys.modules['processing_module'] = _pm

from openCV.processing_module import process_frame, Obstacle
from openCV.main import print_obstacle_info, print_detected_group_info


'''
Currently testing picking up an object from picking bay 1
'''
# TARGETS FOR TESTING
# TARGET_STATION_INDEX = 3
# TARGET_ITEM = warehouseObjects.bowl

# TARGET_BAY = (4,2) 

# CONSTANTS
STATION_ARRIVE_DIST = 0.17
STATION_DEPART_DIST = 0.3
RAMP_ARRIVE_DIST = 0.7
BEARING_LIMIT = 0.05
STATION_ROW_DIST = 1.2
SHELF_GAP = 0.3
BAY_GAP = 0.27
STATION_DIST = 0.14

SIM_TICK = 0.05 # 50ms

# SIMULATED ORDER
orders = [(3, (1,2)), (2, (3,0)), (1, (5,3)) ]
ORDER_INDEX = 0

def processOrder():
    currentOrder = orders[ORDER_INDEX]
    TARGET_STATION_INDEX = currentOrder[0]
    TARGET_BAY = currentOrder[1]
    return TARGET_STATION_INDEX, TARGET_BAY



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


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# CONFIGURE SCENE PARAMETERS
sceneParameters = SceneParameters()

sceneParameters.pickingStationContents[0] = warehouseObjects.bowl
sceneParameters.pickingStationContents[1] = warehouseObjects.mug
sceneParameters.pickingStationContents[2] = warehouseObjects.bottle

sceneParameters.obstacle0_StartingPosition = -1 #[-0.2, -0.25]
sceneParameters.obstacle1_StartingPosition = -1  # Use current position
sceneParameters.obstacle2_StartingPosition = -1  # Use current position

# CONFIGURE ROBOT PARAMETERS
robotParameters = RobotParameters()

robotParameters.driveType = 'differential'        # Type of drive system
robotParameters.minimumLinearSpeed = 0.0          # Minimum forward speed (m/s)
robotParameters.maximumLinearSpeed = 0.3         # Maximum forward speed (m/s)
robotParameters.driveSystemQuality = 1            # Drive quality (0-1, 1=perfect)

robotParameters.cameraOrientation = 'landscape'   # Camera orientation
robotParameters.cameraDistanceFromRobotCenter = 0.1  # Distance from robot center (m)
robotParameters.cameraHeightFromFloor = 0.15      # Height above floor (m)
robotParameters.cameraTilt = 0.0                  # Camera tilt angle (radians)

# Object detection ranges (in meters)
robotParameters.maxItemDetectionDistance = 1.0         # Items
robotParameters.maxPackingBayDetectionDistance = 5   # Picking stations
robotParameters.maxObstacleDetectionDistance = 1.5     # Obstacles
robotParameters.maxRowMarkerDetectionDistance = 5    # Row markers
robotParameters.minWallDetectionDistance = 5

# Item collection settings
robotParameters.collectorQuality = 1              # Collector reliability (0-1)
robotParameters.maxCollectDistance = 0.15         # Maximum collection distance (m)


# sync mode

robotParameters.sync = False
STEP_DT = 0.01  # 10 ms physics tick

def tick(n=1):
    for _ in range(n):
        robot.sim.step()  

# simulator based timesteps

def sim_wait(dt_s: float):
    t0 = robot.sim.getSimulationTime()
    while robot.sim.getSimulationTime() - t0 < dt_s:
        time.sleep(0.001)


def updateBearings():
    robot.UpdateObjectPositions()
    objectsRB = robot.GetDetectedObjects([
        warehouseObjects.items,
        warehouseObjects.shelves,
        warehouseObjects.row_markers,
        warehouseObjects.obstacles,
        warehouseObjects.pickingStation,
        warehouseObjects.PickingStationMarkers
    ])

    return objectsRB

def calcPickingStationRB(obstacles, pickingStationRBS):
    debugOpenCV = True
    # TODO: solve issue with openCV detecting 4 markers
    picking_stations = Obstacle.filter_by_type(obstacles, "Picking Station")
    pickingStationRB = {1: None, 2: None, 3: None}
    if debugOpenCV:
        pickingStationRB[1] = pickingStationRBS[0]
        pickingStationRB[2] = pickingStationRBS[1]
        pickingStationRB[3] = pickingStationRBS[2]
    else:
        if picking_stations:
            for station in picking_stations:
                if pickingStationRBS[station.station_id-1] != None:
                # TODO: change this to be detected distance with openCV processing for final code
                    pickingStationRB[station.station_id] = (pickingStationRBS[station.station_id-1][0], station.bearing[0])
    return pickingStationRB

def calcShelfMarkersRB(obstacles, rowMarkerRBS):
    shelfMarkers = Obstacle.filter_by_station_id(obstacles,1)
    shelfMarkerRB = {1: None, 2: None, 3: None}
    if shelfMarkers:
        for shelf in shelfMarkers:
            if shelf.type_name != "Picking Station":
                if rowMarkerRBS[shelf.station_id-1] != None:
                    shelfMarkerRB[shelf.station_id] = (shelf.bearing[0], rowMarkerRBS[shelf.station_id-1][0])
    return shelfMarkerRB

OMEGA_UNIT_DEG = 475
TURNING = False

def _ang_thr_from_deg_s(deg_s):
    return max(-1.0, min(1.0, deg_s / OMEGA_UNIT_DEG))

def _flush_zero(ms=80):
    t_end = time.perf_counter() + ms/1000.0
    while time.perf_counter() < t_end:
        robot.SetTargetVelocities(0.0, 0.0)
        time.sleep(0.005)
        #tick(int(round(0.005/STEP_DT)))

def turn_deg(angle_deg, speed_deg_s=90.0, ramp_ms=120, zero_hold_ms=120):
    # Block the loop and own the motor pipe
    global TURNING
    TURNING = True
    _flush_zero(80)

    sign = 1.0 if angle_deg >= 0 else -1.0
    thr = _ang_thr_from_deg_s(abs(speed_deg_s))
    dur = abs(angle_deg) / max(abs(speed_deg_s), 1e-6)

    steps = max(1, int(ramp_ms/10))  # 10 ms ticks
    # ramp up
    for k in range(1, steps+1):
        robot.SetTargetVelocities(0.0, sign * thr * (k/steps))
        time.sleep(0.01)
        #tick(int(round(0.01/STEP_DT)))

    # cruise (reserve time for ramp down)
    cruise = max(0.0, dur - steps*0.01*2)
    if cruise > 0:
        robot.SetTargetVelocities(0.0, sign * thr)
        time.sleep(cruise)
        #tick(int(round(cruise/STEP_DT)))

    # ramp down
    for k in range(steps, -1, -1):
        robot.SetTargetVelocities(0.0, sign * thr * (k/steps))
        time.sleep(0.01)
        #tick(int(round(0.01/STEP_DT)))

    _flush_zero(zero_hold_ms)
    TURNING = False



def drive_distance_time(dist_m, linear_cmd=0.10, ramp_ms=120, zero_hold_ms=120, seconds_for_1m=2.0):
    if abs(dist_m) < 1e-6 or abs(linear_cmd) <= 0: return
    global TURNING
    TURNING = True
    _flush_zero(80)

    sign = 1.0 if dist_m >= 0 else -1.0
    dur  = abs(dist_m) * seconds_for_1m          
    steps, tick_s = max(1, int(ramp_ms/10)), 0.01

    for k in range(1, steps+1):
        robot.SetTargetVelocities(sign * abs(linear_cmd) * (k/steps), 0.0)
        time.sleep(tick_s)
        #tick(int(round(tick_s/STEP_DT)))

    cruise = max(0.0, dur - steps*tick_s*2)
    if cruise > 0:
        robot.SetTargetVelocities(sign * abs(linear_cmd), 0.0)
        time.sleep(cruise)
        #tick(int(round(cruise/STEP_DT)))

    for k in range(steps, -1, -1):
        robot.SetTargetVelocities(sign * abs(linear_cmd) * (k/steps if steps else 0.0), 0.0)
        time.sleep(tick_s)
        #tick(int(round(tick_s/STEP_DT)))

    _flush_zero(zero_hold_ms)
    TURNING = False


def steerToBearing(b, limit=BEARING_LIMIT, yaw=0.1):
    if b > limit:
        set_cmd(0, abs(yaw))
        return False
    elif b < -limit:
        set_cmd(0, -abs(yaw))
    else:
        set_cmd(0,0)
        return True



# motor lock helpers

MOTOR_LOCK = False
def set_cmd(v,w):
    if MOTOR_LOCK:
        return
    robot.SetTargetVelocities(v,w)

def with_motor_lock(fn, *args, **kw):
    global MOTOR_LOCK
    MOTOR_LOCK = True
    try:
        return fn(*args, **kw)
    finally:
        robot.SetTargetVelocities(0,0)
        time.sleep(0.2)
        #tick(int(round(0.2/STEP_DT)))
        MOTOR_LOCK = False


def rowIndex(TARGET_BAY):
    TARGET_SHELF_INDEX = TARGET_BAY[0]
    if TARGET_SHELF_INDEX > -1 and TARGET_SHELF_INDEX < 2:
        TARGET_ROW_INDEX = 1
    elif TARGET_SHELF_INDEX > 1 and TARGET_SHELF_INDEX < 4:
        TARGET_ROW_INDEX = 2
    elif TARGET_SHELF_INDEX > 3 and TARGET_SHELF_INDEX < 6:
        TARGET_ROW_INDEX = 3
    return TARGET_ROW_INDEX



def bayTurn(TARGET_BAY):
    TARGET_SHELF_INDEX = TARGET_BAY[0]
    if TARGET_SHELF_INDEX % 2 == 0:
        direction = 'left'
    else:
        direction = 'right'

    return direction


    



if __name__ == '__main__':
    try:
        robot = COPPELIA_WarehouseRobot(robotParameters, sceneParameters, coppelia_server_ip='127.0.0.1', port=23000)

        robot.StartSimulator()

        robotMode = RobotMode.SEARCH_STATION

        show_debug_info = False

        

        if DEBUG_TEST:
            turn_deg(90)

        if not DEBUG_TEST:
            prevMode = robotMode
            while True:
                print("Current mode:", robotMode.name)
                print("Current order:", orders[ORDER_INDEX])



                resolution, image_data = robot.GetCameraImage()
                # packingStation is entire picking station object
                itemsRBS, packingStationRBS, obstaclesRBS, rowMarkerRBS, shelfRBS, pickingStationRBS = updateBearings()
                if packingStationRBS:
                    paDistance, paBearing = packingStationRBS


                if image_data != None:
                    width, height = resolution
                    image_array = np.array(image_data, dtype = np.uint8)
                    image = image_array.reshape((height,width,3))
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    #cv2.imshow('Robot camera', image_bgr)
                    #cv2.waitKey(1)

                    time.sleep(0.1)
                    #tick(int(round(0.1/STEP_DT)))

                    obstacles, trajectory = process_frame(image_bgr)
                    pickingStationRB = calcPickingStationRB(obstacles, pickingStationRBS)
                    shelfMarkerRB = calcShelfMarkersRB(obstacles, rowMarkerRBS)
                    # print("shelfMarkerRB:", shelfMarkerRB)
                    # print_detected_group_info(obstacles)
                    # print_detected_group_info(obstacles)
                    # print_debug_range_bearing("Picking Stations", pickingStationRBS)
                    #print_obstacle_info(obstacles)
                    


                
                if show_debug_info:
                    clear_screen()
                    print("EGB320 Warehouse Robot - Object Detection Status")
                    print("=" * 50)

                    # Display detected objects
                    print_debug_range_bearing("Items", itemsRBS)
                    print_debug_range_bearing("Packing Station", packingStationRBS)
                    print_debug_range_bearing("Obstacles", obstaclesRBS)
                    print_debug_range_bearing("Row Markers", rowMarkerRBS)
                    print_debug_range_bearing("Shelves", shelfRBS)
                    print_debug_range_bearing("Picking Stations", pickingStationRBS)
                    print("=" * 50)


                elif robotMode == RobotMode.DEBUG_STOP:
                    robot.SetTargetVelocities(0,0)

                elif robotMode == RobotMode.SEARCH_STATION:
                    TARGET_STATION_INDEX, TARGET_BAY = processOrder()
                    if packingStationRBS:
                        if paDistance < RAMP_ARRIVE_DIST:
                            aligned = steerToBearing(paBearing)
                            if aligned:
                                robot.SetTargetVelocities(0,0)
                                robotMode = RobotMode.CHECK_ROW
                        else:
                            robot.SetTargetVelocities(0.1,0)
                    else:
                        robot.SetTargetVelocities(0,0.1)
                
                elif robotMode == RobotMode.CHECK_ROW:
                    # TODO: Change to use vision camera instead of virtual
                    if rowMarkerRBS[1] != None: # row 2 is index 1 when using virtual camera
                        row2B = rowMarkerRBS[1][1]
                        aligned = steerToBearing(row2B)
                        if aligned:
                            if rowMarkerRBS[1][0] > STATION_ROW_DIST:
                                robot.SetTargetVelocities(0.1,0)
                            else:
                                robot.SetTargetVelocities(0,0)
                                with_motor_lock(turn_deg, 90)
                                prevMode = robotMode
                                robotMode = RobotMode.TO_RAMP
                    else:
                        robot.SetTargetVelocities(0,-0.1)

                elif robotMode == RobotMode.TO_RAMP:
                    pickingBayDist = TARGET_STATION_INDEX * STATION_DIST
                    with_motor_lock(drive_distance_time, pickingBayDist)
                    with_motor_lock(turn_deg, 90)
                    robotMode = RobotMode.ASCEND_RAMP  

                
                elif robotMode == RobotMode.ASCEND_RAMP:
                    pickingStation = pickingStationRB[TARGET_STATION_INDEX] # TODO: change back to be w/o -1 when openCV fixed
                    if pickingStation:
                        pickingStationBearing = pickingStation[1] 
                        pickingStationDistance = pickingStation[0]
                        if pickingStationDistance > STATION_ARRIVE_DIST:
                            aligned = steerToBearing(pickingStationBearing)
                            if aligned:
                                robot.SetTargetVelocities(0.1,0)
                        else:
                            robot.SetTargetVelocities(0,0)
                            robotMode = RobotMode.COLLECT_ITEM
                    else:
                        robot.SetTargetVelocities(0, 0.1)

                elif robotMode == RobotMode.COLLECT_ITEM:
                    robot.SetTargetVelocities(0,0)
                    success, station = robot.CollectItem(closest_picking_station=True)
                    if success:
                        robotMode = RobotMode.DESCEND_RAMP
                    else:
                        robotMode = RobotMode.ASCEND_RAMP

                elif robotMode == RobotMode.DESCEND_RAMP:
                    pickingStation = pickingStationRB[TARGET_STATION_INDEX]
                    if pickingStation:
                        pickingStationBearing = pickingStation[1] # TODO: change to 0 for openCV
                        pickingStationDistance = pickingStation[0] # TODO: change to 1 for openCV
                        if pickingStationDistance < STATION_DEPART_DIST:
                            robot.SetTargetVelocities(-0.1,0)
                        else:
                            robot.SetTargetVelocities(0,0)
                            robotMode = RobotMode.SEARCH_SHELF
                    else:
                        robot.SetTargetVelocities(-0.1,0)

                
                elif robotMode == RobotMode.SEARCH_SHELF:
                    TARGET_ROW_INDEX = rowIndex(TARGET_BAY)
                    if TARGET_STATION_INDEX == 3:
                        with_motor_lock(turn_deg, 90)
                        if TARGET_ROW_INDEX == 1:
                            with_motor_lock(turn_deg, 90)
                            robotMode = RobotMode.TO_SHELF
                        elif TARGET_ROW_INDEX == 2:
                            with_motor_lock(drive_distance_time, SHELF_GAP)
                            with_motor_lock(turn_deg,90)
                            robotMode = RobotMode.TO_SHELF
                        elif TARGET_ROW_INDEX == 3:
                            with_motor_lock(drive_distance_time, SHELF_GAP*2)
                            robotMode = RobotMode.TO_SHELF

                    elif TARGET_STATION_INDEX == 2:
                        if TARGET_ROW_INDEX == 1:
                            with_motor_lock(turn_deg, -90)
                            with_motor_lock(drive_distance_time, SHELF_GAP/2)
                            with_motor_lock(turn_deg, -90)
                            robotMode = RobotMode.TO_SHELF
                        elif TARGET_ROW_INDEX == 2:
                            with_motor_lock(turn_deg, 90)
                            with_motor_lock(drive_distance_time, 2*SHELF_GAP/3)
                            with_motor_lock(turn_deg, 90)
                            robotMode = RobotMode.TO_SHELF
                        elif TARGET_ROW_INDEX == 3:
                            with_motor_lock(turn_deg, 90)
                            with_motor_lock(drive_distance_time, SHELF_GAP*1.5)
                            with_motor_lock(turn_deg, 90)
                            robotMode = RobotMode.TO_SHELF
                    
                    elif TARGET_STATION_INDEX == 1:
                        if TARGET_ROW_INDEX == 1:
                            with_motor_lock(turn_deg, -90)
                            with_motor_lock(drive_distance_time, (2*SHELF_GAP)/3)
                            with_motor_lock(turn_deg, -90)
                            robotMode = RobotMode.TO_SHELF
                        elif TARGET_ROW_INDEX == 2:
                            with_motor_lock(turn_deg, 90)
                            with_motor_lock(drive_distance_time, SHELF_GAP/3)
                            with_motor_lock(turn_deg, 90)
                            robotMode = RobotMode.TO_SHELF
                        elif TARGET_ROW_INDEX == 3:
                            with_motor_lock(turn_deg, 90)
                            with_motor_lock(drive_distance_time, SHELF_GAP)
                            with_motor_lock(turn_deg, 90)
                            robotMode = RobotMode.TO_SHELF

                
                elif robotMode == RobotMode.TO_SHELF:
                    TARGET_BAY_INDEX = TARGET_BAY[1]
                    # TODO: Change to use vision camera instead of virtual
                    if rowMarkerRBS[TARGET_ROW_INDEX-1] != None: # row 2 is index 1 when using virtual camera
                        rowB = rowMarkerRBS[TARGET_ROW_INDEX-1][1]
                        distIndex = -1*TARGET_BAY_INDEX + 3
                        aligned = steerToBearing(rowB)
                        if distIndex+1 == 1:
                            bayDistance = (1/4)*BAY_GAP
                        else:
                            bayDistance = (1/2)*BAY_GAP + distIndex * BAY_GAP
                        if aligned:
                            if rowMarkerRBS[TARGET_ROW_INDEX-1][0] > bayDistance:
                                robot.SetTargetVelocities(0.1,0)
                            else:
                                robot.SetTargetVelocities(0,0)
                                robotMode = RobotMode.DROP_ITEM
                    else:
                        robot.SetTargetVelocities(0,-0.1)

                

                elif robotMode == RobotMode.DROP_ITEM:
                    if rowMarkerRBS[TARGET_ROW_INDEX-1] != None:
                        shelfB = rowMarkerRBS[TARGET_ROW_INDEX-1][1]
                        aligned = steerToBearing(shelfB)
                        if aligned:
                            direction = bayTurn(TARGET_BAY)
                            if direction == 'right':
                                with_motor_lock(turn_deg, -90)
                            elif direction == 'left':
                                with_motor_lock(turn_deg, 90)
                            with_motor_lock(drive_distance_time, 0.1)
                            robot.SetTargetVelocities(0,0)
                            robot.DropItemInClosestShelfBay()
                            with_motor_lock(drive_distance_time, -0.05)
                            direction = bayTurn(TARGET_BAY)
                            if direction == 'right': angle = 90
                            elif direction == 'left': angle = -90
                            with_motor_lock(turn_deg, angle)
                            aligned = steerToBearing(shelfB)
                            if aligned:
                                robotMode = RobotMode.EXIT_ROW


                elif robotMode == RobotMode.EXIT_ROW:
                    if rowMarkerRBS[TARGET_ROW_INDEX-1] != None:
                        rowD = rowMarkerRBS[TARGET_ROW_INDEX-1][0]
                        if rowD < 5*STATION_ROW_DIST/8:
                            time.sleep(0.1)
                            robot.SetTargetVelocities(-0.1,0)
                        else:
                            robot.SetTargetVelocities(0,0)
                            robotMode = RobotMode.FIND_ORIGIN
                    else:
                        robot.SetTargetVelocities(0, 0.1)

                elif robotMode == RobotMode.FIND_ORIGIN:
                    with_motor_lock(turn_deg, -120)
                    with_motor_lock(drive_distance_time, (SHELF_GAP)*(-TARGET_ROW_INDEX+3))
                    if ORDER_INDEX == 2:
                        robotMode = RobotMode.DEBUG_STOP
                    else:
                        ORDER_INDEX += 1
                        robotMode = RobotMode.SEARCH_STATION



                

    except KeyboardInterrupt:
        print("\nStopping simulation...")
        robot.StopSimulator()
        print("Simulation stopped successfully. Goodbye!")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Stopping simulation...")
        traceback.print_exc()
        try:
            robot.StopSimulator()
        except:
            pass
        print("Please check your CoppeliaSim setup and try again.")


