'''Image processing module for obstacle detection'''
import math
import cv2
import numpy as np
from itertools import combinations

class ColourProfile:
    '''Defines a colour profile in HSV space'''
    def __init__(self, name, h_min, h_max, s_min, s_max, v_min, v_max, colour_bgr):
        self.name = name
        self.lower = np.array([h_min, s_min, v_min])
        self.upper = np.array([h_max, s_max, v_max])
        self.colour_bgr = colour_bgr

# --- Module-level configuration ---
# colour profiles
GREEN = ColourProfile("Green", 45, 92, 55, 255, 20, 255, (0, 255, 0))  # Probably scale up Hmin
ORANGE = ColourProfile("Orange", 0, 21, 100, 255, 50, 255, (0, 140, 255)) 
BLUE = ColourProfile("Blue", 100, 150, 160, 255, 20, 255, (255, 0, 0))
BLACK = ColourProfile("Black", 0, 180, 0, 200, 0, 80, (0, 0, 255))
LOWER_RED = ColourProfile("Red", 0, 0, 120, 255, 120, 255, (0, 0, 255))
UPPER_RED = ColourProfile("Red", 165, 179, 120, 255, 120, 255, (0, 0, 255))
WHITE = ColourProfile("White", 0, 255, 0, 255, 100, 255, (255, 255, 255))
WALL = ColourProfile("Wall",32,255,0,200,175,255,(255,255,255)) 
YELLOW = ColourProfile("Yellow",25,35,100,255,0,255,(0,255,255))
COLOURS = [ORANGE, GREEN,LOWER_RED, UPPER_RED, BLACK, YELLOW, BLUE, WALL]

# Object type constants
TYPE_ROW_MARKER = "Row Marker"
TYPE_PICKING_STATION = "Picking Station"
TYPE_CUBE = "Cube"
TYPE_BALL = "Soccer Ball"
TYPE_BLOCK = "Block"
TYPE_MUG = "Mug"
TYPE_OIL = "Oil Bottle"
TYPE_BOWL = "Bowl"
TYPE_YELLOW_STATION = "Yellow Station"
TYPE_OBSTACLE_FLOOR = "Floor Obstacle"
TYPE_OBSTACLE_TALL = "Tall Obstacle"
TYPE_SHELF = "Shelf"

# Object detection parameters
CIRCULARITY_THRESHOLD = 0.81  # For detecting circular objects
ASPECT_RATIO_THRESHOLD = 1.5  # For distinguishing wide vs tall objects
HEIGHT_WIDTH_RATIO_THRESHOLD = 1.8  # For very tall objects like oil bottle

# Real-world widths in meters
REAL_SIZES = {
    TYPE_ROW_MARKER: 0.065,  # Black circular marker
    TYPE_PICKING_STATION: 0.047,  # Black square marker
    TYPE_CUBE: 0.04,        # Orange cube
    TYPE_BALL: 0.045,        # Soccer ball
    TYPE_BLOCK: 0.065,        # Rectangular block 
    TYPE_MUG: 0.048,         # Mug with handle
    TYPE_OIL: 0.068,          # Oil bottle height
    TYPE_BOWL: 0.04,         # Bowl diameter
    TYPE_YELLOW_STATION: 0.46,
    TYPE_OBSTACLE_FLOOR: 0.35,
    TYPE_OBSTACLE_TALL: 0.05
}

# Processing parameters
KERNEL = np.ones((5, 5), np.uint8)
FOV_WIDTH_DEG = 31.3
FOV_HEIGHT_DEG = 41
FOV_WIDTH_RAD = np.deg2rad(FOV_WIDTH_DEG)
FOV_HEIGHT_RAD = np.deg2rad(FOV_HEIGHT_DEG)
FOCAL_LENGTH_PX_DISTANCE = 1545
FOCAL_LENGTH_PX_BEARING = 1200
WALL_FOCAL_LENGTH_PX_DISTANCE = 1500
CAMERA_HEIGHT_M = 0.103

class Obstacle:
    '''Class representing a detected obstacle'''
    def __init__(self, type_name, centre, area, bearing, distance, shape, colour_bgr, contour=None, station_id=None):
        self.type_name = type_name
        self.centre = centre
        self.area = area
        self.bearing = bearing
        self.distance = distance
        self.shape = shape
        self.colour_bgr = colour_bgr
        self.contour = contour  # Store the actual contour points
        self.station_id = station_id

    def as_dict(self):
        '''Return obstacle attributes as a dictionary'''
        return {
            "type": self.type_name,
            "centre": self.centre,
            "area": self.area,
            "bearing": self.bearing,
            "distance": self.distance,
            "shape": self.shape,
            "station_id": self.station_id
        }

    @staticmethod
    def count_by_type(obstacles):
        """
        Count the number of obstacles by their type
        Args:
            obstacles: List of Obstacle objects
        Returns:
            dict: Dictionary with obstacle types as keys and their counts as values
        """
        counts = {}
        for obstacle in obstacles:
            counts[obstacle.type_name] = counts.get(obstacle.type_name, 0) + 1
        return counts

    @staticmethod
    def filter_by_type(obstacles, type_name):
        """
        Filter obstacles by type
        Args:
            obstacles: List of Obstacle objects
            type_name: Type of obstacles to filter for (e.g., 'Orange', 'Green', etc.)
        Returns:
            list: List of Obstacle objects of the specified type
        """
        return [obs for obs in obstacles if obs.type_name == type_name]

    @staticmethod
    def filter_by_station_id(obstacles, station_id):
        """
        Filter obstacles by station ID
        Args:
            obstacles: List of Obstacle objects
            station_id: Station ID to filter for
        Returns:
            list: List of Obstacle objects with the specified station ID
        """
        return [obs for obs in obstacles if obs.station_id == station_id]


# Processing parameters
KERNEL = np.ones((5, 5), np.uint8)
CIRCLE_PARAM = 0.83
MIN_AREA = 900
SHELF_SIZE_THRESHOLD = 1000

# --- Main processing functions ---
# --- Global marker memory ---
marker_memory = {
    "row_markers": [],        # list of Obstacle objects
    "picking_stations": [],   # list of Obstacle objects
    "frames_since_seen": {}   # maps marker ID to frames since last seen
}

def analyse_shape(contour):
    """analyse shape characteristics of a contour"""
    area = cv2.contourArea(contour)
    if area == 0:
        return None
    
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    extent = float(area) / (w * h) if w * h > 0 else 0
    
    return {
        'area': area,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'width': w,
        'height': h,
        'x': x,
        'y': y
    }

def detect_object_type(contour, colour_name, area, hsv):
    """Determine object type based on shape analysis and colour"""
    shape = analyse_shape(contour)
    if shape is None:
        return None, None
    if colour_name == "Green":
        if shape['height'] / shape['width'] > HEIGHT_WIDTH_RATIO_THRESHOLD:
            return TYPE_OBSTACLE_TALL, "rectangle"
        else:
            return TYPE_OBSTACLE_FLOOR, "rectangle"
    elif colour_name == "Blue":  # Blue shelves don't need specific type
        return TYPE_SHELF, "Rectangle"
    elif colour_name == "Yellow":
        return TYPE_YELLOW_STATION, "rectangle"
    elif colour_name == "Orange":
        # First check for tall objects (oil bottle)
        if shape['height'] / shape['width'] > HEIGHT_WIDTH_RATIO_THRESHOLD:
            return TYPE_OIL, "rectangle"
        
        # Check for circular objects with high circularity
        if shape['circularity'] > 0.82:
            if shape['extent'] > 0.73:  # Soccer ball is very filled
                return TYPE_BALL, "circle"
        
        # Check for cube and block
        if shape['extent'] > 0.87:  # Very high extent for solid shapes
            if abs(shape['aspect_ratio'] - 1.0) < 0.2:  # Very square
                return TYPE_CUBE, "square"
            elif shape['aspect_ratio'] > 1.15:  # Wider than tall
                return TYPE_BLOCK, "rectangle"
        
        if shape['aspect_ratio'] > 1.21:
            if shape['extent'] > 0.75:  # Bowl has medium extent
                return TYPE_BOWL, "circle"
        #print(shape["circularity"])
        #print(shape['aspect_ratio'])
        return TYPE_MUG, "rectangle"
        
    elif colour_name == "Black":
        # Black objects are handled separately in detect_and_group_markers
        return None, None
        
    return None, "unknown"

def calculate_bearing_distance(shape, frame_cx, frame_cy, frame_width, frame_height, real_size=None, obj_type=None, contour=None):
    """
    Calculate bearing (degrees) and distance (meters) to an object.
    Uses contour moments if provided for accurate center.
    """
    # Determine object center
    if contour is not None:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = shape['x'] + shape['width'] // 2
            cy = shape['y'] + shape['height'] // 2
    else:
        cx = shape['x'] + shape['width'] // 2
        cy = shape['y'] + shape['height'] // 2

    dx = cx - frame_cx
    dy = cy - frame_cy

    # Bearing in degrees
    bearing_x = math.degrees(math.atan(dx / FOCAL_LENGTH_PX_BEARING))
    bearing_y = math.degrees(math.atan(dy / FOCAL_LENGTH_PX_BEARING))

    distance = None
    if real_size is not None and obj_type is not None:
        # Pick pixel dimension depending on object type
        if obj_type == TYPE_OIL:  # tall object
            px_size = shape['height']
        elif obj_type in [TYPE_BALL, TYPE_BOWL]:  # circular objects
            px_size = (shape['width'] + shape['height']) / 2
        else:  # default: width
            px_size = shape['width']

        if px_size > 0:
            # Distance using pinhole camera model
            distance = (real_size * FOCAL_LENGTH_PX_DISTANCE) / px_size

    return (bearing_x, bearing_y), distance



def process_frame(frame):
    '''Process a single frame to detect obstacles and trajectory'''
    frame_height, frame_width = frame.shape[:2]
    centrex, centrey = frame_width // 2, frame_height // 2
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (9, 9), 0)
    detected_obstacles = []
    trajectory = []
    
    yellow_mask = cv2.inRange(hsv, YELLOW.lower, YELLOW.upper)
    yellow_present = np.any(yellow_mask > 0)

    if yellow_present:
        # Set HSV min for wall to 200 if yellow is detected
        WALL.lower[2] = 190
    else:
        pass
        WALL.lower[2] = 175

    for colour in COLOURS:
        mask = cv2.inRange(hsv, colour.lower, colour.upper)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if colour.name == "Wall":
        # Find contours of wall mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 500:  # filter out noise
                    continue

                # Visual center of the wall contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centre_x = int(M["m10"] / M["m00"])
                    centre_y = int(M["m01"] / M["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(cnt)
                    centre_x = x + w // 2
                    centre_y = y + h // 2

                # Bottommost pixel for distance calculation
                bottom_y = cnt[:, :, 1].max()
                wall_dist = wall_distance_from_transition(bottom_y, frame_height)

                detected_obstacles.append(
                    Obstacle(
                        type_name="Wall",
                        centre=(centre_x, centre_y),  # center for overlay
                        area=cv2.contourArea(cnt),
                        bearing=(0.0, 0.0),
                        distance=wall_dist,
                        shape="Line",
                        colour_bgr=colour.colour_bgr,
                        contour=cnt
                    )
                )
            continue

        
        # --- Black grouping case ---
        if colour.name == "Black":
            black_groups = detect_and_group_markers(hsv, mask, contours, WALL, centrex, centrey)
            detected_obstacles.extend(black_groups)
            continue
        
        # --- Normal colour contour processing ---
        for cnt in contours:
            shape_info = analyse_shape(cnt)
            if shape_info is None or shape_info['area'] < MIN_AREA:
                continue
                
            if colour.name == "Blue" and shape_info['area'] < SHELF_SIZE_THRESHOLD:
                continue
            
            obj_type, shape_type = detect_object_type(cnt, colour.name, shape_info['area'], hsv)
            if obj_type is None:
                continue
            
            centre = (shape_info['x'] + shape_info['width'],
                      shape_info['y'] + shape_info['height'])
            
            bearing, distance_m = calculate_bearing_distance(
                shape_info, centrex, centrey, frame_width, frame_height,
                REAL_SIZES.get(obj_type),
                obj_type
            )
            
            detected_obstacles.append(
                Obstacle(
                    type_name=obj_type,
                    centre=centre,
                    area=shape_info['area'],
                    bearing=bearing,
                    distance=distance_m,
                    shape=shape_type if shape_type else "unknown",
                    colour_bgr=colour.colour_bgr,
                    contour=cnt
                )
            )
            trajectory.append(centre)

    return detected_obstacles, trajectory

def detect_and_group_markers(hsv, mask, input_contours, white_profile, centrex, centrey):
    """
    Detect and group black markers into picking stations or row markers.
    First determine type per marker using circularity, then calculate distance.
    Group markers afterward, averaging distances for groups.
    """
    detected_groups = []
    black_contours_info = []

    frame_height, frame_width = hsv.shape[:2]

    # Step 1: filter valid black contours and determine type + distance
    for cnt in input_contours:
        area = cv2.contourArea(cnt)
        if area > 500 and has_white_border(hsv, cnt, white_profile):
            # Calculate circularity for type decision
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if circularity > 0.85 and 0.8 < aspect_ratio < 1.2 and y > 0:
                type_name = TYPE_ROW_MARKER
            else:
                type_name = TYPE_PICKING_STATION




            # Bounding rect for distance calculation
            x, y, w, h = cv2.boundingRect(cnt)
            shape_info = {'x': x, 'y': y, 'width': w, 'height': h}

            # Calculate distance with correct marker size
            bearing, distance = calculate_bearing_distance(
                shape_info,
                centrex, centrey,
                frame_width, frame_height,
                REAL_SIZES[type_name],
                type_name
            )

            black_contours_info.append({
                'contour': cnt,
                'shape': shape_info,
                'bearing': bearing,
                'distance': distance,
                'type': type_name
            })

    if not black_contours_info:
        return []

    # Step 2: extract contours for grouping
    black_contours = [info['contour'] for info in black_contours_info]

    # Step 3: group markers into rows using scaled distance
    row_contours, merged_mask = group_same_row_markers_distance(black_contours, hsv.shape)

    # Step 4: assign group distance and bearing
    for row in row_contours:
        # Create mask for this row
        row_mask = np.zeros_like(merged_mask)
        cv2.drawContours(row_mask, [row], -1, 255, -1)

        # Find which contours are in this row
        group_infos = []
        for info in black_contours_info:
            cnt = info['contour']
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if row_mask[cy, cx] == 255:
                    group_infos.append(info)

        if not group_infos:
            continue

        # Average distance and bearing
        avg_distance = sum(info['distance'] for info in group_infos) / len(group_infos)
        avg_bearing_x = sum(info['bearing'][0] for info in group_infos) / len(group_infos)
        avg_bearing_y = sum(info['bearing'][1] for info in group_infos) / len(group_infos)
        avg_bearing = (avg_bearing_x, avg_bearing_y)

        # Bounding box for group
        x, y, w, h = cv2.boundingRect(row)
        centre = (x + w//2, y + h//2)
        area = cv2.contourArea(row)

        # Use the type of first marker in group (all markers same type)
        type_name = group_infos[0]['type']

        detected_groups.append(
            Obstacle(
                type_name=type_name,
                centre=centre,
                area=area,
                bearing=avg_bearing,
                distance=avg_distance,
                shape="group",
                colour_bgr=BLACK.colour_bgr,
                contour=row,
                station_id=len(group_infos)
            )
        )

    # --- Discard picking stations if any row markers exist ---
    row_marker_exists = any(group.type_name == TYPE_ROW_MARKER for group in detected_groups)
    if row_marker_exists:
        detected_groups = [group for group in detected_groups if group.type_name != TYPE_PICKING_STATION]

    return detected_groups

def group_same_row_markers_distance(black_contours, frame_shape):
    """
    Group black markers based on distance rather than morphological dilation.
    Returns group contours and a synthetic mask (like the old version).
    """
    if not black_contours:
        return [], np.zeros(frame_shape[:2], dtype=np.uint8)

    contour_infos = []
    for cnt in black_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        contour_infos.append({
            'contour': cnt,
            'centre': (cx, cy),
            'size': (w + h) / 2
        })

    avg_size = np.mean([info['size'] for info in contour_infos])
    dist_thresh = avg_size * 1.3

    parent = list(range(len(contour_infos)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    for i, j in combinations(range(len(contour_infos)), 2):
        ci, cj = contour_infos[i]['centre'], contour_infos[j]['centre']
        dist = math.dist(ci, cj)
        if dist < dist_thresh:
            union(i, j)

    groups = {}
    for idx, info in enumerate(contour_infos):
        root = find(idx)
        groups.setdefault(root, []).append(info)

    group_contours = []
    merged_mask = np.zeros(frame_shape[:2], dtype=np.uint8)

    for members in groups.values():
        cnts = [m['contour'] for m in members]
        merged = np.vstack(cnts)
        hull = cv2.convexHull(merged)
        group_contours.append(hull)
        cv2.drawContours(merged_mask, [hull], -1, 255, -1)  # fill merged mask

    return group_contours, merged_mask



def has_white_border(hsv, cnt, white_profile, fraction_thresh=0.15):
    contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
    kernel = np.ones((9,9), np.uint8)
    dilated_mask = cv2.dilate(contour_mask, kernel, iterations=1)
    ring_mask = dilated_mask - contour_mask
    colours_around_contour = hsv[ring_mask == 255]
    if colours_around_contour.size == 0:
        return False
    white_pixels = np.all((colours_around_contour >= white_profile.lower) & (colours_around_contour <= white_profile.upper), axis=1)
    white_fraction = np.sum(white_pixels) / len(colours_around_contour)
    return white_fraction >= fraction_thresh

def wall_distance_from_transition(transition_y, img_height):
    """
    Estimate distance to the floor-wall intersection (ground contact point)
    with a horizontally-mounted camera using the pinhole projection model.
    """
    center_y = img_height / 2
    dy = transition_y - center_y  # pixels below optical center

    if dy <= 0:
        # Intersection above or at horizon → invalid geometry
        return None

    # Pinhole model: Z = (H * f) / dy
    distance = (CAMERA_HEIGHT_M * WALL_FOCAL_LENGTH_PX_DISTANCE) / dy
    return distance
