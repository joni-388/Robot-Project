import numpy as np
import pybullet as p
from simulation import Simulation
from typing import Tuple, Literal

class Node:
    """
    Represents a node in a tree.

    Attributes:
        position: The position stored in the node.
        parent: The parent node of the current node.
    """
    def __init__(self, position, 
                 weight: float = 0.0, 
                 qs: np.ndarray = None, 
                 q_dots: np.ndarray = None):
        self.position = position
        self.weight = weight
        self.parent = None


class Planner:
    """
    A class to plan the trajectories in the environment.

    Attributes:
        path: The current planned trajectory path as a list of waypoints.
        smoothened_path: The current smoothed trajectory path as a list of waypoints.
        num_iterations: The max number of iterations to perform during planning.
        step_size: The step size to use during trajectory extension.
        upperlim: the limits of the environment (table)
        lowerlim: the limits of the environment (table)
        robot: The robot object to plan trajectories for.
    """
    
    def __init__(self,
                 num_iterations,
                 step_size,
                 upperlim,
                 lowerlim,
                 robot):
        self.path = None
        self.smoothened_path = None
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.upperlim = upperlim
        self.lowerlim = lowerlim
        self.robot = robot

        pass
    
    
    def plan_ee(self, start, 
                      goal, 
                      obstacles=[], 
                      method: Literal['RRT', 'RRT-Connect'] = 'RRT',
                      *args):
        """
        Plans the end-effector trajectory from the start configuration to the goal configuration.

        Args:
            start: The start configuration of the end-effector.
            goal: The goal configuration of the end-effector.
            obstacles: A list of obstacles in the environment.
            method: The planning method to use. Can be either 'RRT' or 'RRT-Connect'.
            *args: Additional arguments for the planning method.

        Returns:
            The planned trajectory as a list of configurations.

        """
        self.plan_trajectory(start, 
                             goal, 
                             self.num_iterations, 
                             self.step_size, 
                             obstacles, 
                             method=method,
                             smoothing=False)
        return self.path
        
            
    def plan_trajectory(self, 
                        start, 
                        goal, 
                        num_iterations: int, 
                        step_size: float,
                        obstacles: list,
                        method: Literal['RRT', 'RRT-Connect'] = 'RRT',  
                        bias: int = 10,
                        smoothing: bool = False,
                        ):
        """
        Plans a trajectory from the start position to the goal position using the specified method.

        Args:
            start: The start position of the trajectory.
            goal: The goal position of the trajectory.
            num_iterations: The max number of iterations to perform during planning.
            step_size: The step size to use during trajectory extension.
            obstacles: A list of obstacles to avoid during planning.
            method: The planning method to use. Can be either 'RRT' or 'RRT-Connect'.
            bias: The bias value used in vanilla RRT for randomly selecting the goal position.
            smoothing: Whether to apply path smoothing after planning.

        Returns:
            The computed trajectory path as a list of waypoints.
        
        Notes:
            - we use num_iterations to limit the number of iterations and instead continue with
              evasion if no path is found quickly enough.    
        """        
        nodes = [Node(start)]
        
        iter = 0
        if method == 'RRT':
            for _ in range(num_iterations):
            #while True:    
                if iter % bias == 0:
                    rand_point = Node(goal)
                else:    
                    rand_point = self.sample_node()
                iter += 1    
                nearest_node = min(nodes, key=lambda n: float(np.linalg.norm(n.position-rand_point.position)))
                new_node = self.extend(nearest_node, rand_point, step_size)
                new_node.parent = nearest_node
                if not self.check_collisions(new_node, obstacles):
                    nodes.append(new_node)
                    if self.distance(new_node, goal) < step_size:
                        goal_node = Node(goal)
                        goal_node.parent = new_node
                        nodes.append(goal_node)
                        break
                else:
                    continue
        elif method == 'RRT-Connect':
            goal_nodes = [Node(goal)]
            found_goal = False
            for _ in range(num_iterations):
            #while True:  
                if found_goal:
                    break  
                iter += 1
                # extend tree connected to goal nodes to random target
                rand_point = self.sample_node()
                nearest_node = min(goal_nodes, key=lambda n: float(np.linalg.norm(n.position-rand_point.position)))
                # extend(from,to)
                new_node_goal = self.extend(nearest_node, rand_point, step_size)
                new_node_goal.parent = nearest_node
                if not self.check_collisions(new_node_goal, obstacles):
                    goal_nodes.append(new_node_goal)
                    
                    # use new node as target for closest node of other tree
                    nearest_node = min(nodes, key=lambda n: float(np.linalg.norm(n.position-new_node_goal.position)))
                    new_node = self.extend(nearest_node, new_node_goal, step_size)
                    new_node.parent = nearest_node

                    while (not self.check_collisions(new_node, obstacles)) and (not found_goal):
                        nodes.append(new_node)
                        # check if we can connect the trees
                        if self.distance(new_node, new_node_goal.position) < step_size:
                            # merge the trees
                            self.__merge_trees(nodes, new_node, new_node_goal)
                            print('Path found')
                            found_goal = True
                            break
                        else:
                            # if not continue linear extension towards goal tree node
                            temp = new_node
                            new_node = self.extend(new_node, new_node_goal, step_size)
                            new_node.parent = temp        
                else:
                    continue
        if iter == num_iterations:
            print('No path found, continue evasion !')
            return None        
        self.path = self.backtrack(nodes)
        if smoothing:
            self.smoothened_path = self.smooth_path(self.path, degree=4, nbr_eval_points=100)
        print('Planning complete')
        print('Computed path consists of ' + str(len(self.path)) + ' waypoints')    
        return self.path      
        
        
    def sample_node(self) -> Node:
        """
        Samples a new random node in the global bounds of the environment.

        Returns:
            Node: The node object at a random position.

        Notes:
            - We do not allow samples that are too close to the robots base, 
            as the resulting paths would be dangerous for the gripped objects.
        """
        pos = np.random.uniform(self.lowerlim, self.upperlim)
        margin = 0.5
        # define some condiitons for sampling
        # robot init is [0, 0, 0.62], do not sample in robot 
        while not (np.abs(pos[0]) > margin and np.abs(pos[1]) > margin):
            pos = np.random.uniform(self.lowerlim, self.upperlim)
        return Node(pos)
    
     
    def smooth_path(self, path, degree: int = 2, nbr_eval_points: int = 40):
        """
        Smooths the given path using B-spline interpolation.

        Args:
            path (list): The original path to be smoothed.
            degree (int, optional): The degree of the B-spline curve. Defaults to 2.
            nbr_eval_points (int, optional): The number of evaluation points. Defaults to 40.

        Returns:
            list: The smoothed path as a list of positions.

        Notes:
            - If the path is too short for smoothing with the given degree, the function will return the original path.
        """
        nbr_control_points = len(path)
        m = nbr_control_points + 1 + degree
        p = degree - 1
        # check if smooothing i possible based on b-spline degree
        if nbr_control_points < 2*(degree + 1):
            print('Path is too short for smoothening with this degree, continue with unsmoothed path !')
            return path
        
        # unsure here
        knots = np.zeros(m+1)
        knots[(degree):(m-p-1)] = np.linspace(0, 1, m - 2*degree)
        knots[(m-p-2):] = 1
        
        #for i in knots:
        #    print(str(i))
                             
        def b_spline(x, i: int, degree: int = 2):
            # helper function to compute the b-spline basis function
            # https://en.wikipedia.org/wiki/B-spline
            if degree == 0:
                if knots[i] <= x < knots[i+1]:
                    return 1
                else:
                    return 0
            else:
                if knots[i+degree] == knots[i]:
                    a = 0
                else:
                    a = (x - knots[i]) / (knots[i+degree] - knots[i]) * b_spline(x, i, degree-1)
                if knots[i+degree+1] == knots[i+1]:
                    b = 0
                else:
                    b = (knots[i+degree+1] - x) / (knots[i+degree+1] - knots[i+1]) * b_spline(x, i+1, degree-1)
                return  a + b        
        
        def polynomial(x, degree: int = 3):
            # helper function to compute polynomial out of b-spline basis functions
            val = np.zeros(3)
            for i in range(nbr_control_points): 
                val += np.array(path[i]) * b_spline(x,i, degree)  
            return np.array(val) 
        
        #vectorized = np.vectorize(polynomial) !? doesn't work
        new_path = []
        # something is werid here
        eval_points = np.linspace(0, 1, nbr_eval_points + 1)
        for i in eval_points[0:-2]:
            new_path.append(polynomial(i))
        new_path. append(path[-1])    
        print('Path smoothening complete !')
        return new_path
        
                       
    def __merge_trees(self, nodes, 
                      contact_node: Node, 
                      contact_node_goal: Node):
        """
        Merge two trees grown in opposite directions from RRT-Connect into a single tree from start to goal
        by adding the nodes to the tree starting from start node

        Parameters:
        - nodes (list): List of nodes in the tree.
        - contact_node (Node): The node where the two trees make contact (start tree).
        - contact_node_goal (Node): The node where the two trees make contact (goal tree)

        Returns:
        None
        """
        leaf_node = contact_node
        current_node = contact_node_goal.parent
        next_node = current_node.parent
        
        current_node.parent = leaf_node
        nodes.append(current_node)

        # rewiring
        while next_node is not None:
            leaf_node = current_node
            current_node = next_node
            
            next_node = next_node.parent

            current_node.parent = leaf_node
            nodes.append(current_node)
    

    def extend(self, from_node: Node, 
               to_node: Node,    
               step_size: float):
        """
        Extends the current path by creating a new node between the from_node and the to_node with distance
        step_size from the from_node.

        Parameters:
            from_node (Node): The starting node.
            to_node (Node): The target node.
            step_size (float): The step size for extending the path 
            use_steer (bool, optional): Flag to indicate whether to use steering with inverse kinematics. 
                                        Defaults to False.

        Returns:
            Node: The newly created node.

        """
        normal_vector = to_node.position - from_node.position
        normal_vector /= np.linalg.norm(normal_vector)
        new_position = from_node.position + step_size * normal_vector
        new_node = Node(new_position)
        return new_node
    

    def distance(self, node, goal):
        """
        Calculates the Euclidean distance between the given node and the goal position.

        Parameters:
            node (Node): The node whose position is used for distance calculation.
            goal (numpy.ndarray): The goal position.

        Returns:
            float: The Euclidean distance between the node position and the goal position.
        """
        return np.linalg.norm(node.position - goal)
    
    
    def backtrack(self, nodes):
        """
        Backtracks from the last node in the given list of nodes to the start node via parents,
        returns the path as a list of positions.

        Args:
            nodes (list): A list of nodes , where the last node represents the goal node.

        Returns:
            list: A list of positions representing the path from the start position
                  to the goal position.
        """
        path = []
        current = nodes[-1]
        while current is not None:
            path.insert(0, current.position)
            current = current.parent
        return path
    
    
    # function to check end-effector collision with obstacles
    def check_collisions(self, node, obstacles):
        """
        Check if a given node collides with any obstacles using
        the distance between the node and the closest point on the obstacle (plus margin=0.1).

        Parameters:
        - node: The node to check for collisions.
        - obstacles: A list of obstacles to check against.

        Returns:
        - True if a collision is found, False otherwise.
        """
        found_collision = False

        for o in obstacles:
            # obstacle position in ellipsoid (joint) frame
            found_collision = np.linalg.norm(node.position - o.pos) <= (o.radius + 0.1)        
            if found_collision:
                return found_collision
        return found_collision
