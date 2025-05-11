import math

def orientation(p1, p2, p3):
    # Returns 1 if counter-clockwise, -1 if clockwise, 0 if collinear
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    d = (y3 - y2) * (x2 - x1) - (y2 - y1) * (x3 - x2)
    if d > 0:
        return 1
    elif d < 0:
        return -1
    else:
        return 0

def euclidean_dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def gift_wrapping(points):
    if len(points) < 3:
        return points

    on_hull = min(points)  # Start from the leftmost point
    hull = []

    while True:
        hull.append(on_hull)
        next_point = points[0]
        for point in points:
            if point == on_hull:
                continue
            orient = orientation(on_hull, next_point, point)
            if (next_point == on_hull or
                orient == 1 or
                (orient == 0 and euclidean_dist(on_hull, point) > euclidean_dist(on_hull, next_point))):
                next_point = point

        on_hull = next_point
        if on_hull == hull[0]:  # Completed the loop
            break

    return hull


points = [(0, 0), (1, 1), (2, 0), (1, -1), (0, 2), (2, 2)]
hull = gift_wrapping(points)
print("Convex Hull:", hull)