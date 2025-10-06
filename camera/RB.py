import cv2

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

def calcPickingMarkersRB(obstacles):
    pickingStationMarkers = []
    for obstacle in obstacles:
        if obstacle.type_name == "Picking Station":
            pickingStationMarkers.append(obstacle)

    pickingStationsRB = {0: None, 1: None, 2: None}
    if pickingStationMarkers:
        for station in pickingStationMarkers:
            pickingStationsRB[station.station_id-1] = (station.distance, station.bearing[0])

    return pickingStationsRB

'''
Target is name of desired object
'''

def calcObjectRB(obstacles, target):
    objectTypes = ["Cube", "Soccer Ball", "Block", "Mug", "Oil Bottle", "Bowl"]

    for obstacle in obstacles:
        if obstacle.type_name == target:
            objectRB = (obstacle.distance, obstacle.bearing[0])

    return objectRB


def calcPackingStationRB(obstacles):
    for obstacle in obstacles:
        if obstacle.type_name == 'Yellow Station':
            psRB = (obstacle.distance, obstacle.bearing[0])
    return psRB