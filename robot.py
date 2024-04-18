import pybullet as p
import pybullet_robots
import numpy as np
from typing import Tuple, Literal

# # own modules
# from planer import Planer


class Robot:
    """Robot Class.

    The class initializes a franka robot.

    Args:
        init_position: Initial position of the robot.
        orientation: Robot orientation in axis angle representation.
        table_scaling: Scaling parameter for the table.
    """
    def __init__(self,
                 init_position: Tuple[float, float, float] = [0, 0, 0.62],
                 orientation: Tuple[float, float, float] = [0, 0, 0],
                 table_scaling: float = 2.0):

        # load robot
        self.pos = init_position
        self.axis_angle = orientation
        self.tscale = table_scaling
        # to switch back and forth bewteen pose and transaltion control
        self.approach_vel = True

        if self.tscale != 1.0:
            self.pos = [self.pos[0], self.pos[1], self.pos[2] * self.tscale]
        self.ori = p.getQuaternionFromEuler(self.axis_angle)

        self.arm_idx = [0, 1, 2, 3, 4, 5, 6,7,8]#????
        self.default_arm = [-1.8, 0.058, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.gripper_idx = [9, 10]

        self.ee_idx = 11

        self.id = p.loadURDF("franka_panda/panda.urdf", self.pos, self.ori,
                             useFixedBase=True)

        self.lower_limits, self.upper_limits = self.get_joint_limits()

        self.set_default_position()

        for j in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)
            
 

    def set_default_position(self):
        for idx, pos in zip(self.arm_idx, self.default_arm):
            p.resetJointState(self.id, idx, pos)

    def get_joint_limits(self):
        lower = []
        upper = []
        for idx in self.arm_idx:
            joint_info = p.getJointInfo(self.id, idx)
            lower.append(joint_info[8])
            upper.append(joint_info[9])
        return lower, upper

    def print_joint_infos(self):
        num_joints = p.getNumJoints(self.id)
        print('number of joints are: {}'.format(num_joints))
        for i in range(0, num_joints):
            print('Index: {}'.format(p.getJointInfo(self.id, i)[0]))
            print('Name: {}'.format(p.getJointInfo(self.id, i)[1]))
            print('Typ: {}'.format(p.getJointInfo(self.id, i)[2]))

    def get_joint_positions(self):
        states = p.getJointStates(self.id, self.arm_idx)
        #states = p.getJointStates(self.id, range(p.getNumJoints(self.id)))
        # state[0] joint position angle in radians
        # state[1] joint velocity
        # state[2] joint reaction forces (torques)
        # state[3] joint applied forces 
        return [state[0] for state in states]

    def ee_position(self):
        ee_info = p.getLinkState(self.id, self.ee_idx)
        ee_pos = ee_info[0]
        # ee_ori = ee_info[1]
        return ee_pos
    
    def ee_orientation(self):
        ee_info = p.getLinkState(self.id, self.ee_idx)
        ee_ori = ee_info[1]
        return ee_ori

    def position_control(self, target_positions):
        p.setJointMotorControlArray(
            self.id,
            jointIndices=self.arm_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
        )
    
    ############################################    
    ############  Task 1 (Control)  ############
    ############################################
    
    def idk_control(self, x_desired, 
                    alpha: float = 0.1, 
                    method: Literal['Transpose','Pseudoinverse', 'Nullspace'] = 'Transpose',
                    beta: float = 0.1,
                    obstacles: list = [],
                    carrying_object: bool = False):
        """Compute and set velocities based on inverse dynamics control

        Args:
            x_desired (list): 
            alpha (float, optional): Step for gradient descent. Defaults to 0.1.
            method (Literal[Transpose, Pseudoinverse, Nullspace], optional): Desired method to compute velocities. Defaults to 'Transpose'.
        """
        if method not in ['Transpose', 'Pseudoinverse', 'Nullspace']:
            raise ValueError('Invalid method chosen !')
        # get current joint positions
        q = self.get_joint_positions()
        dim = len(q)
        # for desired vel,acc
        zero_vec = [0.0] * dim
        # compute jacobians
        jac_lin, _ = p.calculateJacobian(self.id,
                                               self.ee_idx,
                                               (0.0,0.0,0.0), 
                                               q,
                                               zero_vec,
                                               zero_vec)    
        delta = np.array(x_desired) - np.array(self.ee_position())
        
        # gradient descent depending on chosen method
        if method == 'Nullspace':
            
            def cost_gradient(q, dim):
                """Helper function to compute the cost gradient for the nullspace objective.
                
                    Args:
                        q (list): Current joint positions.
                        dim (int): Dimension of the cost gradient.

                    Returns:
                        numpy.ndarray: The cost gradient.
                    """
                arm_idx = [2, 3, 4, 5, 6, 7, 8]
                # get link states
                link_states = p.getLinkStates(self.id, arm_idx, computeForwardKinematics=True)
                dH = np.zeros(dim)
                # for every link
                for idx in range(len(link_states)):
                    # jacobian for current link
                    jac_lin, _ = p.calculateJacobian(self.id,
                                            idx,
                                            (0.0,0.0,0.0), 
                                            q,
                                            zero_vec,
                                            zero_vec)
                    # for every possible obstacle
                    for o in obstacles:
                        dH += 2 * (np.array(link_states[idx][0]) - o.pos) @ np.array(jac_lin)
                # TODO - add cost for hitting ground ?         
                return dH   
            
            pinv = np.linalg.pinv(np.array(jac_lin))
            # np.eye could be replaced to make movement biased towards certain joints
            dH = cost_gradient(q, dim=dim)
            sensitivity_mat = np.eye(dim)   
            q_dot = alpha * pinv @ delta + beta * (sensitivity_mat - pinv @ np.array(jac_lin)) @ dH
        
        elif method == 'Pseudoinverse':
            # simple implementation of pseudo inverse control
            pinv = np.linalg.pinv(np.array(jac_lin))
            q_dot = alpha * pinv @ delta
        else:
            # even simpler control based on transpose of jacobian
            q_dot = alpha * np.array(jac_lin).T @ delta
        
        # set computed velocities 
        if carrying_object:
            self.close_gripper()   
        p.setJointMotorControlArray(self.id,
                                    jointIndices=self.arm_idx,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=q_dot.tolist())
        
        
    ############################################    
    ############  Task 1 (Evasion)  ############
    ############################################
        
    def evasive_maneuver(self, x_desired, 
                         alpha: float = 0.05, 
                         beta: float = 0.35, 
                         obstacles: list = None,
                         carrying_object: bool = False):  
        """Nullspace control with flipped objectices: the primary objective is to evade obstacles 
           and the secondary objective is to not get too far away from the goal.

        Args:
            x_desired (list): 
            alpha (float, optional): Step for gradient descent (secondary objective). Defaults to 0.1.
            beta (float, optional): Step for gradient descent (primary objective). Defaults to 0.35.
            obstacles (list, optional): List of obstacle spheres. Defaults to empty list.
        """
        assert obstacles is not None, 'No obstacles to avoid !'

        # get current joint positions
        q = self.get_joint_positions()
        dim = len(q)
        # for desired vel,acc
        zero_vec = [0.0] * dim
        # compute jacobians
        jac_lin, jac_ang = p.calculateJacobian(self.id,
                                               self.ee_idx,
                                               (0.0,0.0,0.0), 
                                               q,
                                               zero_vec,
                                               zero_vec)  
        delta = np.array(x_desired) - np.array(self.ee_position())  
        
        def cost_gradient(q, dim):
                """Helper function to compute the cost gradient for the nullspace objective.
                
                    Args:
                        q (list): Current joint positions.
                        dim (int): Dimension of the cost gradient.

                    Returns:
                        numpy.ndarray: The cost gradient.
                    """
                arm_idx = [2, 3, 4, 5, 6, 7, 8]
                # get link states
                link_states = p.getLinkStates(self.id, arm_idx, computeForwardKinematics=True)
                dH = np.zeros(dim)
                # for every link
                for idx in range(len(link_states)):
                    # jacobian for current link
                    jac_lin, _ = p.calculateJacobian(self.id,
                                            idx,
                                            (0.0,0.0,0.0), 
                                            q,
                                            zero_vec,
                                            zero_vec)
                    # for every possible obstacle
                    for o in obstacles:
                        dH += 2 * (np.array(link_states[idx][0]) - o.pos) @ np.array(jac_lin)
                # TODO - add cost for hitting ground ?         
                return dH   
            
        pinv = np.linalg.pinv(np.array(jac_lin))
        # np.eye could be replaced to make movement biased towards certain joints
        dH = cost_gradient(q, dim=dim)  
        
        # evade obstacles with pinv and add nullspace connstraint of not going too far away from path 
        sensitivity = np.eye(dim)
        q_dot = -beta*pinv @ np.array(jac_lin) @ dH \
              + alpha*(sensitivity - pinv @ np.array(jac_lin)) @ pinv @ delta
        
        #q_dot = beta * dH 
        # set computed velocities   
        if carrying_object:
            self.close_gripper(velocity=-0.2, force=2.0) 
        p.setJointMotorControlArray(self.id,
                                    jointIndices=self.arm_idx,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=q_dot.tolist())
        
        
    def check_danger(self, obstacles, obstacle_margin: float = 0.05, arm_margin: float = 0.05):
        """Checks if any part of the robot is too close to obstacles

        Args:
            obstacle_spheres (list): List of obstacle spheres
            obstacle_margin (float, optional): Safety margin, increases radius of obstacles artifivially. 
                                               Defaults to 0.05.
            arm_margin (float, optional): Safety margin, increases radius of ellipses around
                                          robot links artificially. Defaults to 0.05.

        Returns:
            bool: True if any sphere is too close to robot
            
        Notes:
            - uses ghost obstacles by translating obstacle positions along velocity vector
            - checks if randomly sampled points on sphere intersects the ellipsoid defined by the robot links
        """
        # check end-effector
        for o in obstacles:
            if np.linalg.norm(np.array(self.ee_position()) - np.array(o.pos)) <= (o.radius + 0.3):
                return True
        arm_idx = [5, 6, 7, 8] # frst indices cannot collide
        link_states = p.getLinkStates(self.id, arm_idx, computeForwardKinematics=True)
        
        for idx in range(len(link_states)-1):
            # get the link positions and the middle point
            r = np.array(link_states[idx][0])
            q = np.array(link_states[idx+1][0])

            # check collision with every obstacle
            for o in obstacles:
                # check collision of obstacle with frame
                # sample a random vector in 3d space
                for _ in range(20):
                    t = np.random.uniform(-1, 1, 3)
                    t = t / np.linalg.norm(t)
                    # generate random position on surface of the safety sphere
                    point = o.pos + (o.radius + obstacle_margin) * t 
                    # check if point is inside ellipsoid 
                    if np.linalg.norm(r-point) + np.linalg.norm(q-point) <= (np.linalg.norm(r-q) + arm_margin):
                        print('Collision imminent !')
                        return True
                    # if we have velocity estimates available, check with ghosts
                    if o.v is not None:
                        point = o.pos + o.v * (1/240.) + (o.radius + obstacle_margin) * t 
                        if np.linalg.norm(r-point) + np.linalg.norm(q-point) <= (np.linalg.norm(r-q) + arm_margin):
                            print('Collision imminent !')
                            return True
        return False
    
    
    ############################################    
    ############  Task 2 (Grasping) ############
    ############################################
    
    def open_gripper(self, velocity: float = 0.1):
        """Opens the robot's gripper.

        Args:
            velocity (float, optional): Desired velocity of the gripper elements. Defaults to 0.05.
        """
        p.setJointMotorControlArray(self.id,
                                    jointIndices=self.gripper_idx,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[velocity, velocity])
    
    
    def close_gripper(self, velocity: float = -0.2, force: float = 1.0):
        """Closes the robot's gripper.

        Args:
            velocity (float, optional): Desired velocity of the gripper elements. Defaults to -0.05.
        """
        p.setJointMotorControlArray(self.id,
                                    jointIndices=self.gripper_idx,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=[velocity, velocity],
                                    forces=[force, force])
                               
    def approach_grasp(self, grasp, alpha: float = 0.15):
        """Approaches the grasp position.

        Args:
            grasp ([type]): [description]
            velocity (float, optional): [description]. Defaults to 0.05.
        """
        # get current joint positions
        q = self.get_joint_positions()
        dim = len(q)
        # for desired vel,acc
        zero_vec = [0.0] * dim
        jac_lin, jac_ang = p.calculateJacobian(self.id,
                                               self.ee_idx,
                                               (0.0,0.0,0.0), 
                                               q,
                                               zero_vec,
                                               zero_vec)  
        # the grasps are usually a bit too high, so we need to correct that
        hot_fix = np.array(grasp.pose.translation) - np.array([0,0,0.04])
        #hot_fix[2] = 1.28
        delta_lin = hot_fix - np.array(self.ee_position()) 

        # pose error 
        R_goal = np.array(grasp.pose.rotation.as_matrix())
        #np.array(p.getMatrixFromQuaternion(grasp.pose.rotation.as_quat())).reshape(3,3)
        R_current = np.array(p.getMatrixFromQuaternion(self.ee_orientation())).reshape(3,3)
        R_rel = R_goal @ R_current.T 
        
        # compute rotation angle of relative rot
        theta = np.arccos((np.trace(R_rel) - 1) / 2)
        # if abs(theta - np.pi) < 0.005: 
        #     theta = 0
        #     n = np.ones(3)
        # else:    
        #     n = (1/(2 * np.sin(theta))) * np.array([R_rel[2,1] - R_rel[1,2], 
        #                                             R_rel[0,2] - R_rel[2,0], 
        #                                             R_rel[1,0] - R_rel[0,1]]) 
        
        print(np.linalg.norm(delta_lin))
        print(theta)

        if np.linalg.norm(delta_lin) > 0.01:
            if self.approach_vel:   
                pinv_lin = np.linalg.pinv(np.array(jac_lin))      
                
                q_dot = alpha * (pinv_lin @ delta_lin)
                p.setJointMotorControlArray(self.id,
                                            jointIndices=self.arm_idx,
                                            controlMode=p.VELOCITY_CONTROL,
                                            targetVelocities=q_dot.tolist())
                #self.approach_vel = False
                return False
            elif theta > 0.25:
                diff = np.array(p.getEulerFromQuaternion(p.getDifferenceQuaternion(self.ee_orientation(), grasp.pose.rotation.as_quat())))
                pinv_rot = np.linalg.pinv(np.array(jac_ang))
                #q_new = q + 0.05 * (pinv_rot @ (theta * n))
                q_new = 0.7 * alpha * (pinv_rot @ diff)
                p.setJointMotorControlArray(self.id,
                                            jointIndices=self.arm_idx,
                                            controlMode=p.VELOCITY_CONTROL,
                                            targetVelocities=q_new)
                # q_new = q + 0.08 * (pinv_rot @ )
                # p.setJointMotorControlArray(self.id,
                #                             jointIndices=self.arm_idx,
                #                             controlMode=p.POSITION_CONTROL,
                #                             targetPositions=q_new)
                self.approach_vel = True
                return False
            else:
                self.approach_vel = True
                return False
        else:
            return True
    
     