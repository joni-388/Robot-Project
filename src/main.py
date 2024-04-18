import numpy as np
from simulation import Simulation, Camera
import pybullet as p
import time

# own modules
import grasping
from tracker import Tracker 
from planner import Planner
import utils
import matplotlib.pyplot as plt
from enum import Enum


sim = Simulation(cam_pose=np.array([0.0, -0.75, 1.6]),
                 target_pose=np.array([0.0, -0.65, 1.19]),#1.0, 0, 1.7
                 target_object="YcbBanana",
                 randomize=False)
#sim.stop_obstacles()

robot = sim.get_robot()
goal = np.array([0.65, 0.8, 1.6])  # goal on table: np.array([0.65, 0.8, 1.24]) -> offset for robot + 0.2


# Define the bounding box limits
upperlim = np.array([1.5, 1.5, 2.5])
lowerlim = np.array([-0.5, -1.5, 1.3])
#utils.plot_bounding_box(lowerlim,upperlim)

# plot goal position
goal_a = np.array([0.45, 0.55, 1.55])#np.array([0.45, 0.55, 1.55]) 
goal_b = goal_a + 0.02
#utils.plot_bounding_box(goal_a,goal_b,color=[1,1,0])

#plot start postion
start_a = np.array([1.2,-1,1.5])
start_b = start_a + 0.02
#utils.plot_bounding_box(start_a,start_b,color=[0,0,1])


# enum for main objective state machine
class Mode(Enum):
    GRASPING = 1
    RETRACTING = 2
    # after grasping
    NAVIGATING = 3
    # at first contact
    EVADING = 4
    # after first contact
    REACHING = 5
    
current_mode = Mode.GRASPING

class custom_Obstacle():
    def __init__(self, pos, radius, v=np.array([0,0,0])):
        self.pos = pos
        self.radius = radius
        self.v = v
        

obstacel_spheres = []
for o in sim.obstacles:
    position, orientation = p.getBasePositionAndOrientation(o.id)
    
    obstacle = custom_Obstacle(np.array(position),radius=0.065)
    obstacel_spheres.append(obstacle)
    
# helper to test planning without relying on tracking
def get_obstacles(obstacle_spheres):
    obstacle_spheres = []
    for o in sim.obstacles:
        position, orientation = p.getBasePositionAndOrientation(o.id)
        v = p.getBaseVelocity(o.id)
        obstacle = custom_Obstacle(np.array(position),radius=0.065, v=np.array(v))
        obstacle_spheres.append(obstacle)
    return obstacle_spheres
# utils.plot_obstacles(obstacel_spheres)


# add planer
num_iterations = 700
step_size = 0.1
upperlim = np.array([1.5, 1.5, 2.5])
lowerlim = np.array([-0.5, -1.5, 1.3])
start_position = np.array(robot.ee_position())
goal_position = goal_a

# init planner, plan path
planner = Planner(num_iterations=num_iterations,
                  step_size=step_size,
                  upperlim=upperlim,
                  lowerlim=lowerlim,
                  robot=robot)   
path = planner.plan_ee(start=start_position, 
                       goal=goal_position,
                       obstacles=obstacel_spheres,
                       method = 'RRT-Connect')


# sample grasps
grasps = grasping.sample_grasps(sim, None)
# grasp heuristic
best_grasp = None
for g in grasps:
    if best_grasp is None or np.linalg.norm(np.array(g.pose.translation) - robot.ee_position()) < np.linalg.norm(np.array(best_grasp.pose.translation) - robot.ee_position()):
        best_grasp = g
    

# visualize grasps
# for g in grasps:
#     grasp_a = g.pose.translation
#     grasp_b = grasp_a + 0.005
#     utils.plot_bounding_box(grasp_a,grasp_b,color=[1,1,0])
# grasp_a = best_grasp.pose.translation - np.array([0,0,0.04])
# grasp_b = grasp_a + 0.008
# utils.plot_bounding_box(grasp_a,grasp_b,color=[1,0,0])


#### INIT TRACKER ####
tracker = Tracker(sim, tracker_type='KCF')
first_iter = True

# init stuff for full objective
init_position = robot.ee_position()
path_idx = 1
closing_time = 0

for t in range(20000):
    
    ### Tracker, comment in alogn with tracker above to see bounding boxes ###
    # ! there is a sleeping time at init
    if t % 10 == 0:
        obstacle_coordinates_world,obstacle_positions_SECONDCAM, obstacle_positions_FIXEDCAM= tracker.step()
        os_second = [custom_Obstacle(pos,0.1) for pos in obstacle_positions_SECONDCAM]
        os_fixed = [custom_Obstacle(pos,0.1) for pos in obstacle_positions_FIXEDCAM]
        os_world = [custom_Obstacle(pos,0.1) for pos in obstacle_coordinates_world]
        utils.plot_obstacles(os_world,color=[1,0,0])
        if first_iter:
            first_iter = False
            time.sleep(20)
    
    # MAIN OBJECTIVE
    if current_mode == Mode.GRASPING:
        if robot.approach_grasp(best_grasp, alpha = 1.5):
            robot.close_gripper()
            closing_time += 1
            if closing_time > 50:
                current_mode = Mode.RETRACTING
        else:
            robot.open_gripper() 
    elif current_mode == Mode.RETRACTING:
        robot.close_gripper()
        robot.idk_control(init_position, 
                        alpha=0.4, 
                        method='Pseudoinverse',)
        if np.linalg.norm(np.array(init_position) - np.array(robot.ee_position())) < 0.05:
            current_mode = Mode.NAVIGATING
    else:
        # now we need to register the obstacles and follow a trajectory (precomputed)
        ee = np.array(robot.ee_position())
        obstacel_spheres = get_obstacles(obstacle_spheres=obstacel_spheres)
        check = robot.check_danger(obstacel_spheres, obstacle_margin=0.15, arm_margin=0.1) 
        # adapt path node
        if np.linalg.norm(ee - path[path_idx]) < 0.1:
            if path_idx < len(path) - 1:
                path_idx +=1 
        
        # logic for bringing the robot to the goal
        if check:
            # in this case the robot is in danger, use evasive maneuver
            current_mode = Mode.EVADING
            robot.evasive_maneuver(x_desired=goal_position, alpha=0.01, beta=0.25, obstacles=obstacel_spheres, carrying_object=True)
        elif (not check) and current_mode == Mode.EVADING:
            # in this case we have evaded the obstacles, set danger to false and replan
            current_mode = Mode.REACHING
            path = planner.plan_ee(start=robot.ee_position(), 
                                goal=goal_position,
                                method='RRT',
                                obstacles=obstacel_spheres)
            if path is None:
                current_mode = Mode.EVADING
                robot.evasive_maneuver(x_desired=goal_position, alpha=0.05, beta=0.2, obstacles=obstacel_spheres, carrying_object=True)
            else:
                if len(path) > 1:
                    path_idx = 1
                else:   
                    path_idx = 0
                robot.idk_control(path[path_idx], 
                                alpha=0.3, 
                                method='Nullspace',
                                beta=0.2, 
                                obstacles=obstacel_spheres,
                                carrying_object=True,)
        else:  
            # in this case the robot is not in danger, continue with approaching the target
            if current_mode == Mode.REACHING:
                if (np.linalg.norm(ee - goal_position)) < 0.1:
                    robot.open_gripper()
                # if (robot.ee_position()[0] - 0.05) > goal_position[0] and (robot.ee_position()[1] - 0.05) > goal_position[1]:
                #     robot.open_gripper()
                else:   
                    robot.idk_control(path[path_idx], 
                                    alpha=0.3, 
                                    method='Nullspace',
                                    beta=0.2, 
                                    obstacles=obstacel_spheres,
                                    carrying_object=True,)
            else:  
                robot.idk_control(path[path_idx], 
                                alpha=0.3, 
                                method='Pseudoinverse',
                                carrying_object=True,) 

        
    sim.step()
