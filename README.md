# Robot Project

The overall Goal of the project is to grasp a YCB-Object and place it in a goal basket while avoiding obstacles. This Project is implemented in Python with PyBullet on a Franka robot.
To achieve the goal, multiple tasks are implemented:

## Task 1 (Control)

An IK-solver for the Franka-robot is implemented. You can use the pseudoinverse or the transpose based solution. It can be used to move the robot to a certain goal position. It utilizes the `calculateJacobian` from pybullet.

## Task 2 (Grasping)

In order to sample a grasp, the [GIGA](https://github.com/iROSA-lab/GIGA) library is used.

## Task 3 (Localization & Tracking)

The grasp object should be placed in the goal-basket. In order to avoid the obstacles (red spheres), they need to be tracked. A camera is positioned as a sensor tracks the obstacles via a KCF tracker from the open-cv library.

## Task 4 (Planning)

To place the object in the basket, a plan need to be generated. Here, the RRT method and RRT-Connect is implemented and used for a dynamic planner. Furthermore, a null space control is implemented which realizes the obstacle avoidance.

# Result
[Here](demo_with_banana.webm) is a demonstration of the result. 
For further information refer to the report.



Make sure you have [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) installed beforehand.

Then env can be set up by
```shell
conda env create -f environment_project.yml
```

