import numpy as np
from simulation import Camera
from giga.perception import *
from giga.utils.transform import Transform
from giga.grasp_sampler import GpgGraspSamplerPcl
import open3d as o3d
from giga.grasp import Label

def sample_grasps(sim, grasp_generator):
    gripper_finger_depth = 0.05
    size = 6 * gripper_finger_depth
    rgb, depth = sim.get_renders(cam_type=Camera.CUSTOMCAM)
    

    proj_matrix = np.asarray(sim.cam_matrices[Camera.CUSTOMCAM][1]).reshape([4, 4], order="F")
    view_matrix = np.asarray(sim.cam_matrices[Camera.CUSTOMCAM][0]).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / sim.height, -1:1:2 / sim.width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([point_cloud]) # DEBUG
    # filter and downsample points
    point_cloud = point_cloud.voxel_down_sample(0.005)
    # filter to inside the goal volume: sim.target_pose
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(np.array(sim.target_pose) - np.array([0.15, 0.15, 0.15]), np.array(sim.target_pose) + np.array([0.15, 0.15, 0.15]))
    point_cloud = point_cloud.crop(bounding_box)
    # o3d.visualization.draw_geometries([point_cloud]) # DEBUG
    
    # sample grasps
    num_grasps = 15
    num_parallel_workers = 1

    sampler = GpgGraspSamplerPcl(0.045) # Franka finger depth is actually a little less than 0.05
    safety_dist_above_table = gripper_finger_depth # tweak based on how high the grasp should be from the table
    grasps, grasps_pos, grasps_rot = sampler.sample_grasps_parallel(point_cloud, num_parallel=num_parallel_workers, num_grasps=num_grasps, max_num_samples=80,
                                        safety_dis_above_table=safety_dist_above_table, show_final_grasps=False)
    


    print("Found ", len(grasps), " grasps")
    return grasps
    
