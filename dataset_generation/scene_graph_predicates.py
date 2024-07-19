import shapely
import numpy as np
from shapely.geometry import Point

#map -> map
def adjacent_to(src, dest, radius):
    
    # one of the tuples (src, dest) or (dest, src) depending on which one has multiple polygons
    uneven_pair = [i for i in [(src, dest), (dest, src)] if i[0]['layer'] == 'drivable_area']
    if len(uneven_pair) == 1:
        multi, single = uneven_pair[0]
        single_geom_buffered = single['geom'].buffer(radius/2)
        for geom in multi['geoms']:
            geom = geom.buffer(radius/2)
            if geom.intersects(single_geom_buffered):
                return True
        return False
    if len(uneven_pair) >= 2:
        multi1, multi2 = uneven_pair[0]
        for geom1 in multi1['geoms']:
            for geom2 in multi2['geoms']:
                geom1_buffered = geom1.buffer(radius/2)
                geom2_buffered = geom2.buffer(radius/2)
                if geom1_buffered.intersects(geom2_buffered):
                    return True
        return False
    
    src_geom_buffered = src['geom'].buffer(radius/2)
    dest_geom_buffered = dest['geom'].buffer(radius/2)
    return src_geom_buffered.intersects(dest_geom_buffered)

#map -> map
def intersects_with(src, dest):
    uneven_pair = [i for i in [(src, dest), (dest, src)] if i[0]['layer'] == 'drivable_area']
    if len(uneven_pair) == 1:
        multi, single = uneven_pair[0]
        for geom in multi['geoms']:
            if geom.intersects(single['geom']):
                return True
        return False
    if len(uneven_pair) >= 2:
        multi1, multi2 = uneven_pair[0]
        for geom1 in multi1['geoms']:
            for geom2 in multi2['geoms']:
                if geom1.intersects(geom2):
                    return True
        return False

    return src['geom'].intersects(dest['geom'])

#instance -> map
def on(src, dest):
    position = Point(src['pose'][:2])

    if dest['layer'] == 'drivable_area':
        for geom in dest['geoms']:
            if geom.intersects(position):
                return True
        return False

    return position.intersects(dest['geom'])

#ego -> instance
def in_front_of(src, dest, radius):
    x, y = get_relative_pos(src['pose'], dest['pose'][:2])
    pos = np.array([x,y])
    if np.linalg.norm(pos) < radius:
        return y < 0 and abs(x) < abs(y)
    return False

#ego -> instance
def behind(src, dest, radius):
    x, y = get_relative_pos(src['pose'], dest['pose'][:2])
    pos = np.array([x,y])
    if np.linalg.norm(pos) < radius:
        return y > 0 and abs(x) < abs(y)
    return False

#ego -> instance
def right_of(src, dest, radius):
    x, y = get_relative_pos(src['pose'], dest['pose'][:2])
    pos = np.array([x,y])
    if np.linalg.norm(pos) < radius:
        return x < 0 and abs(y) < abs(x)
    return False

#ego -> instance
def left_of(src, dest, radius):
    x, y = get_relative_pos(src['pose'], dest['pose'][:2])
    pos = np.array([x,y])
    if np.linalg.norm(pos) < radius:
        return x > 0 and abs(y) < abs(x)
    return False



#helpers==================

'''
transform dest position into src frame
src facing direction becomes y axis
src right direction becomes x axis
src_pose has yaw, dest_pos doesn't
'''
def get_relative_pos(src_pose, dest_pos):
    pos = np.array(src_pose[:2])
    yaw = src_pose[2]
    dest_pos = np.array(dest_pos)

    forward = np.array([np.cos(yaw), np.sin(yaw)])
    # Right vector (perpendicular to forward)
    right = np.array([np.sin(yaw), -np.cos(yaw)])

    # Step 2: Transform object position to car's coordinate system
    relative_pos = dest_pos - pos
    
    # Project relative position onto forward and right vectors
    x = np.dot(relative_pos, right)
    y = np.dot(relative_pos, forward)

    return x, y