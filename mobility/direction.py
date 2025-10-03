from time import sleep, monotonic, time
import math

WHEEL_DIAMETER_MM = 30
TRACK_WIDTH_MM = 170

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

    board.set_encoder_enable([board.M1, board.M2])
    board.set_encoder_reduction_ratio([board.M1, board.M2], int(gear_ratio))

    # 2) Geometry
    circ = (1/2)*math.pi * (wheel_diameter_mm / 1000.0)           # m
    b = wheelbase_mm / 1000.0                               # m
    theta_rad = abs(angle_deg) * math.pi / 180.0

    target = 0.5 * b * theta_rad                            # m
    slow_down_window_m = 0.5 * b * (slow_down_window_deg * math.pi / 180.0)

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
            t_prev = t_now      

            rpmL, rpmR = board.get_encoder_speed([board.M1, board.M2])

            # Distance this step
            sL += abs(rpmL) * dt / 60.0 * circ
            sR += abs(rpmR) * dt / 60.0 * circ

            s_avg = 0.5 * (sL + sR)

            # Watchdog
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
                board.motor_movement([board.M1], oL, cmd_speed)
                board.motor_movement([board.M2], oR, cmd_speed)

    finally:
        board.motor_stop(board.ALL)

    # Report measured angle
    theta_meas_rad = (2.0 * s_avg) / b
    return math.degrees(theta_meas_rad)


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

            revL = abs(rpmL) * dt / 60.0
            revR = abs(rpmR) * dt / 60.0


            sL += revL * circ
            sR += revR * circ

            s = 0.5 * (sL + sR)

            # Watchdog
            if (revL + revR) > 0:
                s_last_change_t = t_now
            elif (t_now - s_last_change_t) > watchdog_s:
                break

            remaining = target - s
            if remaining <= 0:
                break

            if remaining < slow_down_window_m:
                scaled = speed_pct * (remaining / slow_down_window_m)
                cmd_speed = max(min_speed_pct, scaled)
                board.motor_movement([board.M1], board.CCW, speed_pct)
                board.motor_movement([board.M2], board.CW, speed_pct_M2)

    finally:
        board.motor_stop(board.ALL)

    return 0.5 * (sL + sR)


def calcBayDistance(TARGET_BAY):
   reverseBayID = abs(TARGET_BAY-4)
   distance = (reverseBayID-1)*26 + 13
   return distance/100


def steerToBearing(b, limit, board):
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


def rowIndex(TARGET_BAY):
    TARGET_SHELF_INDEX = TARGET_BAY[0]
    if TARGET_SHELF_INDEX > -1 and TARGET_SHELF_INDEX < 2:
        TARGET_ROW_INDEX = 1
    elif TARGET_SHELF_INDEX > 1 and TARGET_SHELF_INDEX < 4:
        TARGET_ROW_INDEX = 2
    elif TARGET_SHELF_INDEX > 3 and TARGET_SHELF_INDEX < 6:
        TARGET_ROW_INDEX = 3
    return TARGET_ROW_INDEX


def turnSearch(speed, board):
    board.motor_movement([board.M1], board.CW, speed)
    board.motor_movement([board.M2], board.CW, speed)