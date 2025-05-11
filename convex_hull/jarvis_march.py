import math

def orientation(p1, p2, p3):
    x1, y1, x2, y2, x3, y3 = *p1, *p2, *p3
    d = (y3-y2)*(x2-x1) - (y2-y1)*(x3-x2)
    if d >0:
       return 1
    elif d < 0:
       return -1
    else:
       return 0


def eucledian_dist(p1, p2):
    x1, y2, x2, y2 = *p1, *p2 
    return math.sqrt((y2-y1)**2 + (x2-x1)**2)



def gift_wrapping(points):
    on_hull = min(points) # start
    hull = []
    while True:
        hull.append(on_hull) # traces the optimal path
        next_point = point[0] # assume at start the next point is the first element in the list of points
        for point in points:
            orient = orientation(on_hull, next_point, point)
            if next_point == on_hull or  # check if next_point is there in the hull as we appended in prev cycle
               orient==1 # counter clockwise
               or (orient == 0 and eucledian_dist(on_hull, point) > (eucledian_dist(on_hull, next_point): # collinear
               next_point = point
            on_hull = next_point
            if on_hull == hull[0] # check whether it has reached the start point again
               break
    return hull
                   
# Ref: https://youtu.be/nBvCZi34F_o?feature=shared
              
      
        
