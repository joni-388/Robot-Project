import cv2
import sys
from simulation import  Camera
import numpy as np 
import matplotlib.pyplot as plt
import time

# for debugging use: 
# # import pdb; pdb.set_trace()
class Tracker:
    """
    Implementation of a Tracker class used to detect and track red spheres in the camera images of the simulation environment.
    """
    
    def __init__(self, sim, tracker_type,PLOT=True):
        """
        Initializes the Tracker object.

        Args:
            sim (Simulation): The simulation object.
            tracker_type (str): The type of tracker to be used. Options are
                                - 'BOOSTING'
                                - 'MIL'
                                - 'KCF'
                                - 'TLD'
                                - 'MEDIANFLOW'
                                - 'GOTURN'
                                - 'MOSSE'
                                - 'CSRT'                       
            PLOT (bool, optional): Flag to enable/disable plotting. Defaults to True.
        """
        self.PLOT=PLOT
        # tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = tracker_type#s[1]
        self.sim = sim
        
        # define the transformation matrix from pixel to world coordinates for the fixed camera
        projection_matrix_FIXEDCAM = np.asarray(self.sim.cam_matrices[Camera.FIXEDCAM][1]).reshape([4, 4], order="F")
        view_matrix_FIXEDCAM = np.asarray(self.sim.cam_matrices[Camera.FIXEDCAM][0]).reshape([4, 4], order="F")
        self.tran_pix_world_FIXEDCAM = np.linalg.inv(np.matmul(projection_matrix_FIXEDCAM, view_matrix_FIXEDCAM))
        
        # define the transformation matrix from pixel to world coordinates for the second camera
        projection_matrix_SECONDCAM = np.asarray(self.sim.cam_matrices[Camera.SECONDCAM][1]).reshape([4, 4], order="F")
        view_matrix_SECONDCAM = np.asarray(self.sim.cam_matrices[Camera.SECONDCAM][0]).reshape([4, 4], order="F")
        self.tran_pix_world_SECONDCAM= np.linalg.inv(np.matmul(projection_matrix_SECONDCAM, view_matrix_SECONDCAM))

        # self.tran_pix_world = {'FIXEDCAM':self.tran_pix_world_FIXEDCAM, 'SECONDCAM':self.tran_pix_world_SECONDCAM}
        self.tran_pix_world = {Camera.FIXEDCAM:self.tran_pix_world_FIXEDCAM, Camera.SECONDCAM:self.tran_pix_world_SECONDCAM}
        
        # image dimensions
        self.image_width = 256 
        self.image_height = 256
        self.far = 5.0
        self.near = 0.05
        
        # for color detection for the balls
        self.red_lower = (0,100,100) 
        self.red_upper = (10,255,255)
        
        # init tracker CUSTOMCAM
        rgb,_ = self.sim.get_renders(cam_type=Camera.FIXEDCAM)
        bboxes = self.detect_balls(rgb)
        self.multiTracker_FIXEDCAM = self.init_trackers(rgb, bboxes,self.tracker_type)
        
        # init tracker SECONDCAM
        rgb,_ = self.sim.get_renders(cam_type=Camera.SECONDCAM)
        bboxes = self.detect_balls(rgb)
        self.multiTracker_SECONDCAM = self.init_trackers(rgb, bboxes,self.tracker_type)
         
        self.multiTrackers = {Camera.FIXEDCAM:self.multiTracker_FIXEDCAM, Camera.SECONDCAM:self.multiTracker_SECONDCAM} 
                
        self.iter_idx = 1
        self.intervall = 10
        self.count_fails = 0
        
        return
    
    
    def step(self):
        """
        Perform a step in the tracking process.

        This method retrieves obstacle coordinates from two different cameras, 
        calculates the world coordinates of the obstacles, and returns the 
        obstacle coordinates in the world, as well as the coordinates from 
        the second camera and the fixed camera.

        Returns:
            obstacle_coordinates_world (list): A list of obstacle coordinates in the world.
            obstacle_coords_SECONDCAM (list): A list of obstacle coordinates from the second camera.
            obstacle_coords_FIXEDCAM (list): A list of obstacle coordinates from the fixed camera.
        """
        obstacle_coords_FIXEDCAM, img_coords_FIXEDCAM = self.step_one_CAM(cam_type=Camera.FIXEDCAM)
        obstacle_coords_SECONDCAM, img_coords_SECONDCAM = self.step_one_CAM(cam_type=Camera.SECONDCAM)

        obstacle_coordinates_world = []
        for i in range(2):
            try:
                p = np.array([obstacle_coords_SECONDCAM[i][0], obstacle_coords_FIXEDCAM[i][1], obstacle_coords_FIXEDCAM[i][2]])
                obstacle_coordinates_world.append(p)
            except:
                pass 

        return obstacle_coordinates_world, obstacle_coords_SECONDCAM, obstacle_coords_FIXEDCAM
   
    
    
    def step_one_CAM(self,cam_type):
        """
        Perform a step in the tracking process for a single camera.
        Computes the bounding boxes of the detected spheres, updates the multi-object tracker,
        and returns the world coordinates and image coordinates of the detected supheres.
        
        
        Parameters:
        - cam_type (str): The type of camera used for capturing images.

        Returns:
        - obstacle_coordinates_world (list): A list of world coordinates of the detected balls.
        - img_coords (list): A list of image coordinates (x, y, depth) of the detected balls.
        """
        rgba, depth = self.sim.get_renders(cam_type=cam_type)
        rgb = rgba.copy()[:,:,0:3] # copy for openCV -> otherwise error in rectangel drawing

        # reinitialize trackers every 10th step to avoid tracking failures
        if self.iter_idx % self.intervall == 0:
            detected_boxes = self.detect_balls(rgb)
            if not  len(detected_boxes) == len(self.multiTrackers[cam_type].getObjects()):
                self.multiTrackers[cam_type] = self.init_trackers(rgb, detected_boxes, self.tracker_type)
            self.iter_idx = 0
        self.iter_idx += 1 
        
        # Update tracker
        success, boxes = self.multiTrackers[cam_type].update(rgb)  # gets rgb beacuse its initialized with rgb also its uses bgr
        
        # if one of the trackers fails, reinitialize all trackers
        if not success:
            self.count_fails += 1
            detected_boxes = self.detect_balls(rgb)
            self.multiTrackers[cam_type] = self.init_trackers(rgb, detected_boxes,self.tracker_type)
            boxes = detected_boxes
        
        
        # get world coordinates of the detected balls
        img_coords = []
        obstacle_coordinates_world = []
        for bbox in boxes:
            # get the center of the bounding box
            x_img = (int)(bbox[0] + np.floor(bbox[2]/2))
            y_img = (int)(bbox[1] + np.floor(bbox[3]/2))
            
            # get the depth value by taking the minimum depth value of the bounding box
            a = (int)(bbox[0])
            b = (int)((bbox[0]+bbox[2]))
            c = (int)(bbox[1])
            d = (int)((bbox[1]+bbox[3]))
            depth_value = depth[a:b,c:d].min()
            # depth_value =  self.far * self.near / (self.far - (self.far - self.near) * depth_value)
            # depth_value = depth_value/depth.max()
            
            pixel_coords = np.array([x_img,y_img])
            world_coords = self.pixel_to_world(pixel_coords, depth_value, self.tran_pix_world[cam_type])
            obstacle_coordinates_world.append(world_coords)
            img_coords.append(np.array([x_img,y_img,depth_value]))
            
            if self.PLOT:
                # draw the bounding box in camera image
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                rgb = rgb.copy()
                cv2.rectangle(rgb, p1, p2, (255,0,0), 2, 1)
                cv2.rectangle(depth, p1, p2, (1), 2, 1)
        
        if self.PLOT:
            depth =  self.far * self.near / (self.far - (self.far - self.near) * depth)
            depth = depth / depth.max() #* 255
            print(depth.max(),depth.min())
            cv2.imshow(f"{cam_type}",rgb)
            cv2.imshow(f"{cam_type}_depth",depth)
            # cv2.moveWindow(f"{cam_type}_depth", 0, 100) 
            cv2.waitKey(1)
            
            # cv2.imshow("Tracking", rgb)
            # cv2.waitKey(1)
        
        return obstacle_coordinates_world,img_coords
    
    
    
    def stop(self):
        pass
    
    
    def detect_balls(self,rgb):
        """
        Detects red spheres in an RGB image.

        Parameters:
        - rgb (numpy.ndarray): The input RGB image.

        Returns:
        - list: A list of bounding boxes (x, y, width, height) of the detected red spheres.

        Raises:
        - ValueError: If no red spheres are detected in the image.
        """
        # convert rgb to hsv
        hsv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # define the range of red color in hsv and create a mask
        frame_threshed = cv2.inRange(hsv_img, self.red_lower, self.red_upper)
        
        # find contours in the mask
        contours,_ = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_ExTERNAL cv2.RETR_TREE
        
        # filter contours by area and store them in a list 
        min_area = 0
        red_balls = []
        for contour in contours:   
            if cv2.contourArea(contour) > min_area:
                red_balls.append(contour)
        
        # get the bounding boxes of the contours
        bboxes = []
        for ball in red_balls:
            (x,y,w,h) = cv2.boundingRect(ball)
            bbox = (x,y,w,h)
            bboxes.append(bbox)
            rgb = rgb.astype(np.uint8)
            cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,255,0),2)
        
        if self.PLOT:
            pass
            # cv2.imshow('red balls',rgb)
            # cv2.waitKey(1)
         
        if len(bboxes) == 0:
            raise ValueError('No spheres detected') 
            
        return bboxes
    


    def pixel_to_world(self, image_point, depth_value, tran_pix_world):
        """
        Converts a pixel coordinate and depth value to world coordinates.

        Args:
            image_point (tuple): The pixel coordinates (x, y) of the image point.
            depth_value (float): The depth value of the image point.
            tran_pix_world (numpy.ndarray): The transformation matrix from pixel coordinates to world coordinates.

        Returns:
            numpy.ndarray: The world coordinates (x, y, z) of the image point.
        """
        # Convert image coordinates to NDC
        ndc_x = 2.0 * image_point[0] / self.image_width - 1.0
        ndc_y = 1.0 - 2.0 * image_point[1] / self.image_height  # Y is inverted in OpenGL NDC

        # Convert depth from [0, 1] to [-1, 1]
        depth_value = 2 * depth_value - 1   

        # Convert NDC to clip coordinates
        clip_coords = np.array([ndc_x, ndc_y, depth_value, 1.0])
        
        # Convert clip coordinates to world coordinates
        world_coords = np.matmul(tran_pix_world, clip_coords.T).T
        world_coords /= world_coords[3]  
        
        return world_coords[:3]
    
    
    @staticmethod
    def init_trackers(rgb, bboxes, tracker_type):
        """
        Initializes and returns a multi-object tracker.

        Parameters:
        - rgb (numpy.ndarray): The RGB image frame.
        - bboxes (list): A list of bounding boxes representing the objects to track.
        - tracker_type (str): The type of tracker to use.

        Returns:
        - multiTracker (cv2.legacy.MultiTracker): The initialized multi-object tracker.

        """
        multiTracker = cv2.legacy.MultiTracker_create()
        for bbox in bboxes:
            multiTracker.add(Tracker.get_tracker(tracker_type), rgb, bbox)
        return multiTracker
    
    
    @staticmethod
    def get_tracker(tracker_type):
        """
        Returns an instance of the OpenCV tracker based on the specified tracker_type.

        Parameters:
        - tracker_type (str): The type of tracker to create. Valid options are:
            - 'BOOSTING': Boosting-based tracker
            - 'MIL': Multiple Instance Learning (MIL) tracker
            - 'KCF': Kernelized Correlation Filters (KCF) tracker
            - 'TLD': Tracking, Learning and Detection (TLD) tracker
            - 'MEDIANFLOW': Median Flow tracker
            - 'GOTURN': Generic Object Tracking Using Regression Networks (GOTURN) tracker
            - 'MOSSE': Minimum Output Sum of Squared Error (MOSSE) tracker
            - 'CSRT': Discriminative Correlation Filter with Channel and Spatial Reliability (CSRT) tracker

        Returns:
        - tracker: An instance of the specified tracker type.

        Raises:
        - ValueError: If the specified tracker_type is not found.
        """
        tracker = None
        
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            # tracker = cv2.TrackerKCF_create()
            tracker = cv2.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
            
        if tracker is None:
            raise ValueError('Tracker type not found')
    
        return tracker

 
 

