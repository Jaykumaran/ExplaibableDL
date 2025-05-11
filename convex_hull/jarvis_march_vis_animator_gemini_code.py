# Code provided by Gemini 2.5 Pro
# I dont own any credit for the code,, Just added here for future reference and visual interpretation about the concept better


import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Point representation: (x, y) tuples

def orientation_detailed(p, q, r):
    """
    Finds the orientation of an ordered triplet (p, q, r).
    Returns:
        0: Collinear points
        1: Clockwise turn from pq to qr (r is to the right of vector pq)
       -1: Counter-clockwise turn from pq to qr (r is to the left of vector pq)
    """
    # Using the formula: (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
    # This formula gives:
    # > 0 if (p,q,r) is Clockwise
    # < 0 if (p,q,r) is Counter-Clockwise
    # = 0 if Collinear
    val = (q[1] - p[1]) * (r[0] - q[0]) - \
          (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0  # Collinear
    return 1 if val > 0 else -1 # Clockwise or Counterclockwise

def distance_sq(p1, p2):
    """Calculates the square of the distance between two points."""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def jarvis_march_trace(points_list):
    """
    Computes the convex hull using Jarvis March and yields states for animation.
    points_list: A list of (x,y) tuples.
    """
    points = list(set(points_list)) # Remove duplicates, convert to list
    n = len(points)

    if n < 3:
        # Yield a final state if n < 3
        yield {
            'all_points': points,
            'hull_points': points[:], # Copy of points, as they form the "hull"
            'current_pivot': None,
            'candidate_next': None,
            'checking_point': None,
            'status': f"Hull is points themselves (n={n}<3)." if points else "No points.",
            'final_hull_path': points[:] + ([points[0]] if len(points) == 2 else []) # close path for 2 points
        }
        return

    # Find the starting point: the one with the smallest y (and smallest x if tie)
    start_index = 0
    for i in range(1, n):
        if points[i][1] < points[start_index][1]:
            start_index = i
        elif points[i][1] == points[start_index][1] and points[i][0] < points[start_index][0]:
            start_index = i
    
    hull = []
    current_pivot_idx = start_index
    
    while True:
        hull.append(points[current_pivot_idx])
        
        yield {
            'all_points': points,
            'hull_points': list(hull),
            'current_pivot': points[current_pivot_idx],
            'candidate_next': None, # About to search for the next one
            'checking_point': None,
            'status': f"Point {points[current_pivot_idx]} added to hull. Searching next...",
            'final_hull_path': None
        }

        candidate_next_idx = (current_pivot_idx + 1) % n # Pick an initial candidate (any point other than current_pivot)
                                                    # Make sure it's not current_pivot_idx itself if n is small
        if candidate_next_idx == current_pivot_idx:
            # This can happen if all points are collinear and we picked the wrong end, or n is small
            # Try a different initial candidate
            for i in range(n):
                if i != current_pivot_idx:
                    candidate_next_idx = i
                    break
            if candidate_next_idx == current_pivot_idx: # Still same, means only one unique point or all points identical
                 yield {
                    'all_points': points, 'hull_points': list(hull), 'current_pivot': points[current_pivot_idx],
                    'candidate_next': None, 'checking_point': None, 'status': "Cannot find distinct next point.",
                    'final_hull_path': list(hull) + ([hull[0]] if len(hull) > 1 else [])
                }
                 return


        yield {
            'all_points': points,
            'hull_points': list(hull),
            'current_pivot': points[current_pivot_idx],
            'candidate_next': points[candidate_next_idx],
            'checking_point': None, # About to iterate through others
            'status': f"Initial candidate for next: {points[candidate_next_idx]}",
            'final_hull_path': None
        }

        for i in range(n):
            if i == current_pivot_idx: # Don't check against self
                continue

            checking_point = points[i]
            yield { # State before checking orientation
                'all_points': points,
                'hull_points': list(hull),
                'current_pivot': points[current_pivot_idx],
                'candidate_next': points[candidate_next_idx],
                'checking_point': checking_point,
                'status': f"Checking {checking_point} against candidate {points[candidate_next_idx]}",
                'final_hull_path': None
            }

            o = orientation_detailed(points[current_pivot_idx], points[candidate_next_idx], checking_point)

            # If checking_point is more counter-clockwise (o == -1)
            # or if collinear (o == 0) and checking_point is farther than current candidate_next
            if o == -1 or \
               (o == 0 and distance_sq(points[current_pivot_idx], checking_point) > distance_sq(points[current_pivot_idx], points[candidate_next_idx])):
                
                old_candidate_info = f"(was {points[candidate_next_idx]})" if points[candidate_next_idx] != checking_point else ""
                candidate_next_idx = i
                yield { # State after finding a better candidate
                    'all_points': points,
                    'hull_points': list(hull),
                    'current_pivot': points[current_pivot_idx],
                    'candidate_next': points[candidate_next_idx], # New candidate
                    'checking_point': checking_point, # This point caused the change
                    'status': f"New best candidate: {points[candidate_next_idx]} {old_candidate_info}. Turn was {'CCW' if o == -1 else 'Collinear (farther)'}",
                    'final_hull_path': None
                }
            else: # Candidate did not change
                 yield {
                    'all_points': points,
                    'hull_points': list(hull),
                    'current_pivot': points[current_pivot_idx],
                    'candidate_next': points[candidate_next_idx],
                    'checking_point': checking_point, # This point was checked
                    'status': f"{checking_point} is not 'lefter'. Candidate {points[candidate_next_idx]} remains. Turn was {'CW' if o == 1 else 'Collinear (closer/same)'}",
                    'final_hull_path': None
                }


        current_pivot_idx = candidate_next_idx

        if current_pivot_idx == start_index: # Wrapped around to the start
            break
    
    # Yield final state with completed hull
    final_hull_path = list(hull) + [hull[0]] # Close the polygon for drawing
    yield {
        'all_points': points,
        'hull_points': list(hull),
        'current_pivot': points[current_pivot_idx], # Should be the start point
        'candidate_next': None,
        'checking_point': None,
        'status': f"Hull complete! Found {len(hull)} points.",
        'final_hull_path': final_hull_path
    }
    
    
# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(8, 8))

# Scatter plot for all points (will be static once drawn)
scatter_all_points = ax.scatter([], [], c='blue', s=50, label="All Points")
# Line plot for the hull points found so far
hull_line_plot, = ax.plot([], [], 'r-', lw=2, label="Convex Hull")
# Line plot for current pivot to candidate_next
candidate_line_plot, = ax.plot([], [], 'g--', lw=1.5, label="Pivot-to-Candidate")
# Line plot for current pivot to checking_point
checking_line_plot, = ax.plot([], [], 'k:', lw=1, label="Pivot-to-Checking")
# Highlight points
pivot_marker, = ax.plot([], [], 'o', ms=12, mec='orange', mfc='None', mew=2, label="Current Pivot")
candidate_marker, = ax.plot([], [], '*', ms=12, mec='green', mfc='None', mew=2, label="Candidate Next")
checking_marker, = ax.plot([], [], 'x', ms=10, color='gray', mew=2, label="Checking Point")

# Text annotation for status
status_text_ax = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=9, 
                         bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7))


def init_animation(all_initial_points):
    min_x = min(p[0] for p in all_initial_points) - 1
    max_x = max(p[0] for p in all_initial_points) + 1
    min_y = min(p[1] for p in all_initial_points) - 1
    max_y = max(p[1] for p in all_initial_points) + 1
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize='small', loc='lower right')
    ax.set_title("Jarvis March (Gift Wrapping) Visualization")

    x_coords = [p[0] for p in all_initial_points]
    y_coords = [p[1] for p in all_initial_points]
    scatter_all_points.set_offsets(list(zip(x_coords, y_coords)))
    
    hull_line_plot.set_data([], [])
    candidate_line_plot.set_data([], [])
    checking_line_plot.set_data([], [])
    pivot_marker.set_data([],[])
    candidate_marker.set_data([],[])
    checking_marker.set_data([],[])
    status_text_ax.set_text("Initializing...")
    
    return (scatter_all_points, hull_line_plot, candidate_line_plot, checking_line_plot, 
            pivot_marker, candidate_marker, checking_marker, status_text_ax)


def update_animation(frame_data):
    hull_pts = frame_data['hull_points']
    current_pivot = frame_data['current_pivot']
    candidate_next = frame_data['candidate_next']
    checking_point = frame_data['checking_point']
    status = frame_data['status']
    final_hull_path = frame_data['final_hull_path']

    # Update hull line
    if final_hull_path: # If final hull is available, draw it closed
        hx = [p[0] for p in final_hull_path]
        hy = [p[1] for p in final_hull_path]
        hull_line_plot.set_color('purple') # Change color for final
        hull_line_plot.set_linewidth(3)
    elif len(hull_pts) >= 2:
        hx = [p[0] for p in hull_pts]
        hy = [p[1] for p in hull_pts]
        hull_line_plot.set_color('red')
        hull_line_plot.set_linewidth(2)
    elif len(hull_pts) == 1: # Draw just the first point if only one
        hx = [hull_pts[0][0]]
        hy = [hull_pts[0][1]]
    else:
        hx, hy = [], []
    hull_line_plot.set_data(hx, hy)

    # Update pivot marker
    if current_pivot:
        pivot_marker.set_data([current_pivot[0]], [current_pivot[1]])
    else:
        pivot_marker.set_data([],[])

    # Update candidate line and marker
    if current_pivot and candidate_next:
        candidate_line_plot.set_data([current_pivot[0], candidate_next[0]], 
                                     [current_pivot[1], candidate_next[1]])
        candidate_marker.set_data([candidate_next[0]],[candidate_next[1]])

    else:
        candidate_line_plot.set_data([], [])
        candidate_marker.set_data([],[])

    # Update checking line and marker
    if current_pivot and checking_point:
        checking_line_plot.set_data([current_pivot[0], checking_point[0]], 
                                    [current_pivot[1], checking_point[1]])
        checking_marker.set_data([checking_point[0]],[checking_point[1]])
    else:
        checking_line_plot.set_data([], [])
        checking_marker.set_data([],[])
        
    status_text_ax.set_text(status)

    return (scatter_all_points, hull_line_plot, candidate_line_plot, checking_line_plot,
            pivot_marker, candidate_marker, checking_marker, status_text_ax)


# --- Main Execution ---
if __name__ == '__main__':
    # Example points
    # S = [(0, 3), (1, 1), (2, 2), (4, 4), (0, 0), (1, 2), (3, 1), (3, 3)]
    S = [(2,2), (4,1), (3,4), (5,3), (1,5), (6,5), (4,6), (2,7), (5,0)]
    # S = [(0,0), (1,0), (2,0), (0,1), (1,1), (0,2)] # Test with some collinear boundary
    # S = [(0,0), (1,1)] # Test n < 3
    # S = [(0,0),(1,1),(2,2),(3,3)] # Test all collinear

    if not S:
        print("Point set is empty. Exiting.")
        exit()
        
    trace_steps = list(jarvis_march_trace(S))

    if not trace_steps:
        print("Jarvis March trace did not produce any steps. This might be due to very few points or an issue.")
        exit()

    # The init_func needs the initial set of points to set axis limits correctly
    # We can pass S directly or get it from the first frame of trace_steps
    initial_points_for_init = trace_steps[0]['all_points'] if trace_steps else S

    ani = animation.FuncAnimation(fig, 
                                  update_animation, 
                                  frames=trace_steps, 
                                  init_func=lambda: init_animation(initial_points_for_init), # Use lambda to pass args
                                  blit=True, # Try blit=True for smoother animation
                                  interval=700, # Milliseconds between frames
                                  repeat=False) # Don't repeat the animation

    plt.tight_layout()
    plt.show()

    # Optional: Print the final hull
    if trace_steps and trace_steps[-1]['final_hull_path']:
        final_hull = trace_steps[-1]['hull_points']
        print("\nFinal Convex Hull Points (in order):")
        for pt in final_hull:
            print(pt)
    elif trace_steps and len(trace_steps[-1]['hull_points']) > 0:
         final_hull = trace_steps[-1]['hull_points']
         print("\nHull Points (potentially incomplete if error):")
         for pt in final_hull:
            print(pt)
