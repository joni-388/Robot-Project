import numpy as np
import pybullet as p

def plot_link_centers(robot):
    """
    Visualize a robots links using a small bounding box.

    Parameters:
    - robot: The robot object 

    Returns:
    None
    """
    link_states = p.getLinkStates(robot.id, robot.arm_idx, computeForwardKinematics=True)
    for idx in range(len(link_states)-1):
        # get the link position and orientation
        r = np.array(link_states[idx][0])
        q = np.array(link_states[idx+1][0])
        
        temp_point = r + 0.025*np.ones(3)
        plot_bounding_box(r, temp_point, color=[1,1,0])
        temp_point = q + 0.025*np.ones(3)
        plot_bounding_box(q, temp_point, color=[1,1,0])


def plot_link_ellipsoids(robot):
    """
    Visualize ellipsoids along the links of a robot arm by sampling points
    on them and plotting a small bounding box.

    Parameters:
    - robot: The robot object 

    Returns:
    None
    """
    # get the link states
    link_states = p.getLinkStates(robot.id, robot.arm_idx, computeForwardKinematics=True)
    for idx in range(len(link_states)-1):
        # get the link position and orientation
        r = np.array(link_states[idx][0])
        q = np.array(link_states[idx+1][0])
                
        nbr_pts = 0
        while nbr_pts < 70:
            # sample random points
            x = np.random.uniform(r[0]-0.2, q[0]+0.2)
            y = np.random.uniform(r[1]-0.2, q[1]+0.2)
            z = np.random.uniform(r[2]-0.2, q[2]+0.2)
            point = np.array([x, y, z])
            
            if np.linalg.norm(r-point) + np.linalg.norm(q-point) <= (np.linalg.norm(r-q) + 0.1):
                # plot the point
                temp_point = point + 0.005*np.ones(3)
                plot_bounding_box(point, temp_point, color=[1,1,0])
                nbr_pts += 1
        

def plot_bounding_box(lowerlim,upperlim,color=[1, 0, 0],linewidth = 2, lifeTime=0):
    """
    Plot a bouning box in the pybullet GUI.

    Parameters:
    - lowerlim: The lower limits of the bounding box
    - upperlim: The upper limits of the bounding box
    - color: The color of the bounding box
    - linewidth: The linewidth of the bounding box 

    Returns:
    None
    """
    # Define the eight points of the bounding box
    points = [
        [lowerlim[0], lowerlim[1], lowerlim[2]],
        [lowerlim[0], upperlim[1], lowerlim[2]],
        [upperlim[0], upperlim[1], lowerlim[2]],
        [upperlim[0], lowerlim[1], lowerlim[2]],
        [lowerlim[0], lowerlim[1], upperlim[2]],
        [lowerlim[0], upperlim[1], upperlim[2]],
        [upperlim[0], upperlim[1], upperlim[2]],
        [upperlim[0], lowerlim[1], upperlim[2]]
    ]
    # Define the lines to be drawn by specifying pairs of points
    lines = [
        (points[0], points[1]), (points[1], points[2]), (points[2], points[3]), (points[3], points[0]),
        (points[4], points[5]), (points[5], points[6]), (points[6], points[7]), (points[7], points[4]),
        (points[0], points[4]), (points[1], points[5]), (points[2], points[6]), (points[3], points[7])
    ]
    # Plot each line
    for line in lines:
        p.addUserDebugLine(line[0], line[1], color, linewidth, lifeTime=lifeTime)   
    return




def plot_obstacles(obstacle_spheres, color=[0, 1, 1], linewidth=5, lifeTime=1):
    """
    Plot obstacles as bounding boxes around the obstacle spheres.

    Parameters:
    - obstacle_spheres: list of obstacle spheres
    - color: RGB color value for the bounding box (default: [0, 1, 1])
    - linewidth: width of the bounding box lines (default: 5)
    - lifeTime: lifetime of the bounding box visualization (default: 1)

    Returns:
    None
    """
    for o, c in zip(obstacle_spheres, [[0, 1, 1], [1, 0, 1]]):
        upperlim = o.pos + o.radius
        lowerlim = o.pos - o.radius
        plot_bounding_box(lowerlim, upperlim, color=c, linewidth=3, lifeTime=lifeTime)
    
    return

