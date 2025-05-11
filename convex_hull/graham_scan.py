import math

# Better than Jarvis in terms of complexity

def orientation(p1, p2, p3):
    # slope logic (m1 - m2) is +ve or -ve or 0
    # +ve - counterclockwise
    # -ve - clockwiser
    # 0 - collinear 
    x1, y1, x2, y2, x3, y3 = *p1, *p2, *p3
    d = (y3-y2)*(x2-x1) - (y2-y1)*(x3-x2)
    if d >0:
       return 1
    elif d < 0:
       return -1
    else:
       return 0


def euclidean_dist(p1, p2):
    x1, y1, x2, y2 = *p1, *p2
    return math.sqrt((y2-y1)**2 + (x2-x1)**2)

# choose a min y point as ref and calculate the angle in horizontal to all the points 
def polar_angle(p1, p2):
    if p1[1] == p2[1]: # 180 deg , both points y are same
       return math.pi
    dy = p1[1] - p2[1]
    dx = p1[0] - p2[0]
    return math.atan2(dy,dx)


def graham_scan(points):
    p0 = min(points, key = lambda p: (p[1], p[0])) # min interms of first y and x as well
    points.sort(key = lambda p: (polar_angle(p0, p), euclidean_dist(p0, p)))
    hull = []
    for i in range(len(points)):
        # we need atleast three point stop to proceed with checking for candidate point
        while len(hull) >= 2 and orientation(hull[-2], hull[-1], points[i]) != 1: # before_last, last point, current_point
              hull.pop() # if clockwise or collinear (!=1) remove that and go back to the stable last point
        hull.append(points[i])
    return hull
      
Ref: https://youtu.be/SBdWdT_5isI?feature=shared
