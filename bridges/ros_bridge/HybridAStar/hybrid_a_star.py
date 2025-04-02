"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from matplotlib.animation import FuncAnimation, FFMpegWriter

from dynamic_programming_heuristic import calc_distance_heuristic
from ReedsSheppPath import reeds_shepp_path_planning as rs
from scipy.spatial.transform import Rotation as Rot
import json
import os
import time
from math import cos, sin, tan, pi
from utils.angle import rot_mat_2d
import json
import os
import multiprocessing

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost

show_animation = True

class Node:

    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions, vels, steers,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.vel_list = vels
        self.steer_list = steers
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost

class Path:

    def __init__(self, x_list, y_list, yaw_list, direction_list, vels, steers, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.direction_list = direction_list
        self.cost = cost
        self.vels = vels
        self.steers = steers

class Config:

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)

class Car_class:
    def __init__(self, config_planner):
        self.maxSteerAngle = np.deg2rad(config_planner['vehicle-params']['max_steer'])
        self.maxVel = config_planner['vehicle-params']['max_vel']
        self.minVel = config_planner['vehicle-params']['min_vel']
        self.N_vel = config_planner['HA*-params']['vel_prec']
        self.N_steer = config_planner['HA*-params']['steer_prec'] # number of steering inputs = 2*steerPresion + 1 within [-maxSteerAngle, maxSteerAngle]
        self.wheelBase = config_planner['vehicle-params']['wheelbase_length']
        self.axleToBack = config_planner['vehicle-params']['axle_to_back']
        self.axleToFront = self.wheelBase + self.axleToBack # assuming space between front axle and front of car = axle to back
        self.length = self.axleToFront + self.axleToBack
        self.width = config_planner['vehicle-params']['vehicle_width']
        self.safety_margin = config_planner['HA*-params']['safety_margin']
        self.bubble_dist = self.wheelBase / 2.0  # distance from rear to center of vehicle.
        self.bubble_r = 2*np.hypot((self.axleToFront) / 2.0, self.width / 2.0)
        # vehicle rectangle vertices
        self.VRX = [self.axleToFront, self.axleToFront, -self.axleToBack, -self.axleToBack, self.axleToFront]
        self.VRY = [self.width / 2, -self.width / 2, -self.width / 2, self.width / 2, self.width / 2]

    def check_car_collision(self, x_list, y_list, yaw_list, ox, oy, kd_tree):
        for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
            carRadius = self.wheelBase/2  # along longitudinal axis

            cx = i_x + carRadius * cos(i_yaw)
            cy = i_y + carRadius * sin(i_yaw)
            
            carRadius_query = self.length
            ids = kd_tree.query_ball_point([cx, cy], carRadius_query)

            if not ids:
                continue

            if not self.rectangle_check(i_x, i_y, i_yaw,
                                    [ox[i] for i in ids], [oy[i] for i in ids]):
                return False  # collision

        return True  # no collision


    def rectangle_check(self, x, y, yaw, ox, oy):
        # transform obstacles to base link frame
        # rot = rot_mat_2d(yaw)
        ##  three-circle distance
        cx = x + (self.wheelBase/2) * math.cos(yaw)
        cy = y + (self.wheelBase/2) * math.sin(yaw)

        h, w = self.length / 2, self.width / 2  # Half-length and half-width of vehicle
        wi = self.safety_margin  # Radius of pedestrian model

        offsets = np.array([-1, 0, 1])  # Three points along vehicle's length

        # Compute expanded positions for the ego vehicle (3 points)
        ego_x = cx + offsets * h * np.cos(yaw)  # Shape: (3,)
        ego_y = cy + offsets * h * np.sin(yaw)  # Shape: (3,)
        
        ox = np.array(ox)
        oy = np.array(oy)

        # Compute pairwise distances using broadcasting
        dist_matrix = np.sqrt((ego_x[:, None] - ox[None, :]) ** 2 +
                            (ego_y[:, None] - oy[None, :]) ** 2) - (w + wi)  # Shape: (3, N)

        # Find the minimum distance for each pedestrian
        min_dist = np.min(dist_matrix)  # Shape: (N,)

        # collision = min_dist > 0

        # for iox, ioy in zip(ox, oy):
        #     tx = iox - cx
        #     ty = ioy - cy
        #     # converted_xy = np.stack([tx, ty]).T @ rot
        #     dx = tx * math.cos(yaw) + ty * math.sin(yaw) # from center to obstacle along longitudinal axis
        #     dy = -tx * math.sin(yaw) + ty * math.cos(yaw) # from center to obstacle along lateral axis

        #     # rx, ry = converted_xy[0], converted_xy[1]

        #     if abs(dx) < (self.length/2 + self.safety_margin) and abs(dy) < (self.width / 2 + self.safety_margin):
        #         return False # collision

        #     # if not (rx > (self.axleToFront+self.safety_margin) or rx < (-self.axleToBack-self.safety_margin) or 
        #     #         ry > (self.width / 2.0+self.safety_margin) or ry < (-self.width/ 2.0-self.safety_margin)):
        #     #     return False  # collision

        return min_dist > 0  # no collision
    
    def plot_arrow(self, x, y, yaw, ax, length=1.0, width=0.5, fc="r", ec="k"):
        """Plot arrow."""
        if not isinstance(x, float):
            for (i_x, i_y, i_yaw) in zip(x, y, yaw):
                self.plot_arrow(i_x, i_y, i_yaw)
        else:
            ax.arrow(x, y, length * cos(yaw), length * sin(yaw),
                    fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

    def plot_arrow_return(self, x, y, yaw, ax, length=1.0, width=0.5, fc="r", ec="k"):
        """Plot arrow."""
        if not isinstance(x, float):
            for (i_x, i_y, i_yaw) in zip(x, y, yaw):
                self.plot_arrow(i_x, i_y, i_yaw, ax)
        else:
            # arrow_plot = ax.arrow(x, y, length * cos(yaw), length * sin(yaw),
            #           fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)
            arrow_plot = ax.annotate("", xy=(x + length * cos(yaw), y + length * sin(yaw)),
                                            xytext=(x, y), arrowprops=dict(arrowstyle="simple"))

        return arrow_plot

    def plot_car(self, x, y, yaw, ax):
        car_color = '-g'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0]+x)
            car_outline_y.append(converted_xy[1]+y)

        # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        arrow_x, arrow_y, arrow_yaw = x, y, yaw
        self.plot_arrow(arrow_x, arrow_y, arrow_yaw, ax)

        ax.plot(car_outline_x, car_outline_y, car_color)

    def plot_car_return(self, x, y, yaw, ax):
        car_color = '-g'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0]+x)
            car_outline_y.append(converted_xy[1]+y)

        # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        arrow_x, arrow_y, arrow_yaw = x, y, yaw
        arrow_plot = self.plot_arrow_return(arrow_x, arrow_y, arrow_yaw, ax)

        car_plot, = ax.plot(car_outline_x, car_outline_y, car_color)

        return car_plot, arrow_plot


    def plot_car_trans(self, x, y, yaw, ax):
        car_color = '-g'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0]+x)
            car_outline_y.append(converted_xy[1]+y)

        # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        # arrow_x, arrow_y, arrow_yaw = x, y, yaw
        # plot_arrow(arrow_x, arrow_y, arrow_yaw, ax)

        ax.plot(car_outline_x, car_outline_y, car_color, alpha=0.3)

    def plot_other_car(self, x, y, yaw, ax):
        car_color = '-r'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0]+x)
            car_outline_y.append(converted_xy[1]+y)

        # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        arrow_x, arrow_y, arrow_yaw = x, y, yaw
        self.plot_arrow(arrow_x, arrow_y, arrow_yaw, ax)

        ax.plot(car_outline_x, car_outline_y, car_color, alpha=0.5)

    def plot_other_car_trans(self, x, y, yaw, ax):
        car_color = '-r'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0]+x)
            car_outline_y.append(converted_xy[1]+y)

        # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        # arrow_x, arrow_y, arrow_yaw = x, y, yaw
        # plot_arrow(arrow_x, arrow_y, arrow_yaw, ax)

        ax.plot(car_outline_x, car_outline_y, car_color, alpha=0.2)

    def plot_other_car_return(self, x, y, yaw, ax):
        car_color = '-r'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', -yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0]+x)
            car_outline_y.append(converted_xy[1]+y)

        # arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        arrow_x, arrow_y, arrow_yaw = x, y, yaw
        arrow_plot = self.plot_arrow_return(arrow_x, arrow_y, arrow_yaw, ax)

        car_plot, = ax.plot(car_outline_x, car_outline_y, car_color, alpha=0.5)

        return car_plot, arrow_plot

def map_lot_place_cars(p_x, p_y, p_yaw, indices, Car_obj, p_w, n_s1, obstacleX, obstacleY, center_spots, axes):

    car = np.array(
        [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
         [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
                          [math.sin(p_yaw), math.cos(p_yaw)]])

    car = np.dot(rotationZ, car)

    for j in range(n_s1):

        p_y_j  = p_y + j * p_w

        center_spots.append(np.array([p_x, p_y_j]))

        if j in indices:

            car1 = car + np.array([[p_x], [p_y_j]])  # (2xN) N are vertices
            for ax in axes:
                ax.plot(car1[0, :], car1[1, :], color='black', alpha=0.6)
                ax.plot(p_x, p_y_j, marker='o', color='black', alpha=0.6)

            ## Car
            for i in range(car1.shape[1]-1):
                line = np.linspace(car1[:, i], car1[:, i+1], num = 10, endpoint=True)
                obstacleX = obstacleX + line[:, 0].tolist()
                obstacleY = obstacleY + line[:, 1].tolist()
        else:
            for ax in axes:
                ax.plot(p_x, p_y_j, marker='o', color='grey', alpha=0.6)

    return obstacleX, obstacleY, center_spots

def map_lot(type, config_map, Car_obj, axes):
    """
    ----Inputs----
    type: Type of the map
    config_map: json file that defines the map parameters
    Car_obj: object of class Car_class that defines the dimensions of the car
    axes: list of axes to plot (trajectory, animation)
    """

    obstacleX, obstacleY = [], []
    ## Parallel Parking Map
    y_min = config_map['map-params'][type]['y_min']
    x_min = config_map['map-params'][type]['x_min']
    l_w = config_map['map-params'][type]['lane_width']
    p_w = config_map['map-params'][type]['park_width']
    p_l = config_map['map-params'][type]['park_length']
    n_r = config_map['map-params'][type]['n_rows']
    n_s = config_map['map-params'][type]['n_spaces']  # total number of spaces in one row

    n_s1 = int(n_s / 2) # number of spaces on one side of each row

    ## lot
    # s = config_map['map-params'][type]['start'][:3]
    # s[2] = np.deg2rad(s[2])
    # s[0] = s[0] + l_w + 2*p_l + 0.75*l_w
    # s[1] = s[1] + 0.2*l_w # 0.8*l_w for -np.deg2rad(s[2]) (-90 deg)

    x_max = x_min + (n_r+1)*l_w + 2*n_r*p_l
    y_max = y_min + l_w + n_s1 * p_w + l_w

    ## big_lot: center of vehicle
    # start of ego
    # s = [x_min + (x_max - x_min)/2 - 1.5*p_l, y_max - l_w/4, np.deg2rad(-180.0)] # start_x is middle, start_y is close to y_max
    s = [x_min + 2*l_w + 4*p_l + 1*l_w/4, y_max - l_w - p_w/2, np.deg2rad(-90.0)]
    # s = [x_min + 3*l_w + 4*p_l - 1*l_w/4, y_min + l_w/2, np.deg2rad(90.0)] 

    center_spots = []
    occ_spot_indices = []

    ## x and y coordiantes of the 
    roads_x = [x_min + l_w/2]
    roads_y = [y_min + l_w/2, y_max - l_w/2]

    # Plotting the center lines of each road
    # for ax in axes:
    #     # roads_x
    #     ax.plot([roads_x[0]]*2, [y_min, y_max], color='gold', linestyle='--', alpha=0.6)
    #     # roads_y
    #     ax.plot([x_min, x_max], [roads_y[0]]*2, color='gold', linestyle='--', alpha=0.6) 
    #     ax.plot([x_min, x_max], [roads_y[1]]*2, color='gold', linestyle='--', alpha=0.6) 

    ## Plot the parking lines that differentiate adjacent spots, and generate the center lines of each road
    center_line_park_row_y = [y_min + l_w, y_max - l_w]
    center_line_park_row_x_1 = x_min + l_w + p_l
    for _ in range(n_r):
        roads_x.append(center_line_park_row_x_1 + p_l + l_w/2)
        center_line_park_row_x = [center_line_park_row_x_1] * len(center_line_park_row_y)
        for ax in axes:
            ax.plot(center_line_park_row_x, center_line_park_row_y, color='grey', linestyle='--', alpha=0.6)
            # ax.plot([roads_x[-1]]*2, [y_min, y_max], color='gold', linestyle='--', alpha=0.6) # road lane divider

        short_line_park_row_x = [center_line_park_row_x_1- p_l, center_line_park_row_x_1 + p_l]

        short_line_park_row_y_1 = y_min + l_w + p_w
        for _ in range(n_s1-1):
            short_line_park_row_y = [short_line_park_row_y_1]*len(short_line_park_row_x)
            for ax in axes:
                ax.plot(short_line_park_row_x, short_line_park_row_y, color='grey', linestyle='--', alpha=0.6)

            short_line_park_row_y_1 += p_w

        center_line_park_row_x_1 += 2*p_l + l_w

    """ Initially occupy the parking spots according to a probability distribution 
    the entrance of a mall is at [(x_max - x_min)/2, y_max].
    The occupancy probability decreases as we move away from the entrance
    """
    yaw_r = [np.pi, 0.0] # car facing away from the center parking line of an entire row, aka, reverse-in parking
    # probabilities
    rng = np.random.default_rng(1)
    p_min = config_map['map-params'][type]['occ_prob_min']
    p_max = config_map['map-params'][type]['occ_prob_max']
    p_delta = 0.5 - p_min
    # maximum at middle of the sequence of rows
    p_start_all = [p_min + (i/n_r)*(0.5 - p_min) for i in range(n_r)] + [0.5 - ((i+1)/n_r)*(0.5 - p_min) for i in range(n_r)]
    prob_all_spots = []
    # Iterate through all the parking rows
    for row_split_i in range(int(2*n_r)):
        # probability of occupancy
        p_start = p_start_all[row_split_i]
        p_end = min(p_max, p_start + p_delta)

        p_vertical = np.linspace(p_start, p_end, num=n_s1)
        indices = np.where(rng.binomial(1, p_vertical, size=n_s1))[0] # occupied, where indexing is local to a row

        indices_row = indices + row_split_i*n_s1
        occ_spot_indices = occ_spot_indices + indices_row.tolist() # occupied, where indexing is global to the lot

        # parked cars, where p_x, p_y is the first parking spot of a row
        s_x = x_min + (1 + int(row_split_i / 2)) * l_w + row_split_i * p_l + p_l / 2
        s_y = y_min + l_w + p_w / 2

        p_x = s_x
        p_y = s_y
        p_yaw = yaw_r[row_split_i%2]

        ''''
        1. Place static cars at parking spots given by 'indices', and add them as static obstacles to obstacleX, obstacleY
        2. add (x, y) coordinates of centers of parking spots to center_spots.
        '''
        obstacleX, obstacleY, center_spots = map_lot_place_cars(p_x, p_y, p_yaw, indices, Car_obj, p_w, n_s1, obstacleX, obstacleY, center_spots, axes)
        
        prob_all_spots.append(p_vertical)


    ## manually placing a car
    # car = np.array(
    # [[-Car_obj.length/2, -Car_obj.length/2, Car_obj.length/2, Car_obj.length/2, -Car_obj.length/2],
    #     [Car_obj.width / 2, -Car_obj.width / 2, -Car_obj.width / 2, Car_obj.width / 2, Car_obj.width / 2]])

    # row_split_i_all = [1, 2]
    # j_all = [1, 3]
    # for i in range(len(row_split_i_all)):
    #     row_split_i = row_split_i_all[i] # row index
    #     j = j_all[i] # spot index within the row
    #     p_x = x_min + (1 + int(row_split_i / 2)) * l_w + row_split_i * p_l + p_l / 2
    #     p_y = y_min + l_w + j*p_w / 2
    #     p_yaw = yaw_r[row_split_i%2]
    #     rotationZ = np.array([[math.cos(p_yaw), -math.sin(p_yaw)],
    #                         [math.sin(p_yaw), math.cos(p_yaw)]])
    #     car = np.dot(rotationZ, car)

    #     center_spots.append(np.array([p_x, p_y]))

    #     car1 = car + np.array([[p_x], [p_y]])  # (2xN) N are vertices
    #     for ax in axes:
    #         ax.plot(car1[0, :], car1[1, :], color='black', alpha=0.6)
    #         ax.plot(p_x, p_y, marker='o', color='black', alpha=0.6)

    #     ## Car
    #     for i in range(car1.shape[1]-1):
    #         line = np.linspace(car1[:, i], car1[:, i+1], num = 10, endpoint=False)
    #         obstacleX = obstacleX + line[:, 0].tolist()
    #         obstacleY = obstacleY + line[:, 1].tolist()


    ## Plot occupancy probabilities
    prob_all_spots_n = np.array(prob_all_spots)
    figp, axp = plt.subplots()
    im = axp.imshow(prob_all_spots_n.T, origin='lower')
    axp.set_xticks(np.arange(int(2*n_r)))
    axp.set_yticks(np.arange(n_s1))
    axp.tick_params(axis='both', which='major', labelsize=20)
    cbar = figp.colorbar(im)
    cbar.ax.tick_params(labelsize=20)

    ## Plot the bounds and add them to list of obstacles
    for ax in axes:
        ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color='black')

    for i in np.linspace(x_min, x_max+1, num=100):
        obstacleX.append(i)
        obstacleY.append(y_min)

    for i in np.linspace(y_min, y_max+1):
        obstacleX.append(x_min)
        obstacleY.append(i)
    
    for i in np.linspace(x_min, x_max+1, num=100):
        obstacleX.append(i)
        obstacleY.append(y_max)

    for i in np.linspace(y_min, y_max+1):
        obstacleX.append(x_max)
        obstacleY.append(i)

    obstacleX.append(0.0)
    obstacleY.append(0.0)

    center_spots = np.array(center_spots)

    # for ax in axes:
    #     ax.plot(obstacleX, obstacleY, color='black', alpha=0.6, marker='o', linestyle='')

    return x_min, x_max, y_min, y_max, p_w, p_l, l_w, n_r, n_s, n_s1, obstacleX, obstacleY, s, center_spots, occ_spot_indices, roads_x, roads_y

def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi

class HA_planner:
    def __init__(self, config_planner):
        self.Car = Car_class(config_planner)
        self.xy_resolution = config_planner['HA*-params']['xy_res']
        self.yaw_resolution = np.deg2rad(config_planner['HA*-params']['yaw_res'])
        self.dt = config_planner['HA*-params']['dt']

    def move(self, x, y, yaw, distance, steer):
        x += distance * cos(yaw)
        y += distance * sin(yaw)
        yaw = pi_2_pi(yaw + distance * tan(steer) / self.Car.wheelBase)  # distance/2

        return x, y, yaw


    def calc_motion_inputs(self):
        # steers = np.concatenate((np.linspace(-self.Car.maxSteerAngle, self.Car.maxSteerAngle,
        #                              self.Car.N_steer, endpoint=True), [0.0]))
        # vels = np.linspace(self.Car.minVel, self.Car.maxVel, self.Car.N_vel, endpoint=True)

        # # Create a grid of steer and velocity values
        # steer_grid, vel_grid = np.meshgrid(steers, vels, indexing='ij')

        # # Stack them into an (N, 2) array
        # steers_vels = np.column_stack((steer_grid.ravel(), vel_grid.ravel()))

        # return steers_vels
        for steer in np.concatenate((np.linspace(-self.Car.maxSteerAngle, self.Car.maxSteerAngle,
                                        self.Car.N_steer, endpoint=True), [0.0])):
            for d in [self.Car.minVel, self.Car.maxVel]:
                yield [steer, d]


    def get_neighbors(self, current, config, ox, oy, kd_tree, ax):
    #     steers_vels = self.calc_motion_inputs()

    #    # Compute next nodes for all steer-velocity pairs
    #     nodes = np.array([self.calc_next_node(current, steer, d, config, ox, oy, kd_tree) 
    #                     for steer, d in steers_vels])

    #     # Filter out None values and check validity
    #     valid_nodes = [node for node in nodes if node and self.verify_index(node, config)]

    #     return valid_nodes

        for steer, d in self.calc_motion_inputs():
            node = self.calc_next_node(current, steer, d, config, ox, oy, kd_tree, ax)
            if node and self.verify_index(node, config):
                yield node       


    def calc_next_node(self, current, steer, direction, config, ox, oy, kd_tree, ax):
        x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

        arc_l = self.xy_resolution*1.5
        x_list, y_list, yaw_list, direction_list, vel_list, steer_list = [], [], [], [], [], []
        for _ in np.arange(0, arc_l, self.dt):
            x, y, yaw = self.move(x, y, yaw,  self.dt * direction, steer)
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)
            direction_list.append(direction > 0)
            vel_list.append(direction)
            steer_list.append(steer)

        if not self.Car.check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
            return None

        d = direction > 0
        x_ind = round(x / self.xy_resolution)
        y_ind = round(y / self.xy_resolution)
        yaw_ind = round(yaw / self.yaw_resolution)

        added_cost = 0.0

        if d != current.direction:
            added_cost += SB_COST

        # steer penalty
        added_cost += STEER_COST * abs(steer)

        # steer change penalty
        added_cost += STEER_CHANGE_COST * abs(current.steer - steer)

        cost = current.cost + added_cost + arc_l

        node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                    y_list, yaw_list, direction_list, vel_list, steer_list,
                    parent_index=self.calc_index(current, config),
                    cost=cost, steer=steer)

        return node


    def is_same_grid(self, n1, n2):
        if n1.x_index == n2.x_index \
                and n1.y_index == n2.y_index \
                and n1.yaw_index == n2.yaw_index:
            return True
        return False


    def analytic_expansion(self, current, goal, ox, oy, kd_tree, ax):
        start_x = current.x_list[-1]
        start_y = current.y_list[-1]
        start_yaw = current.yaw_list[-1]

        goal_x = goal.x_list[-1]
        goal_y = goal.y_list[-1]
        goal_yaw = goal.yaw_list[-1]

        max_curvature = math.tan(self.Car.maxSteerAngle) / (self.Car.wheelBase)
        paths = rs.calc_paths(start_x, start_y, start_yaw,
                            goal_x, goal_y, goal_yaw,
                            max_curvature, step_size= self.Car.maxVel*self.dt)

        if not paths:
            return None

        best_path, best = None, None

        for path in paths:
            if self.Car.check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
                cost = self.calc_rs_path_cost(path)
                if not best or best > cost:
                    best = cost
                    best_path = path

        return best_path


    def update_node_with_analytic_expansion(self, current, goal,
                                            c, ox, oy, kd_tree, ax):
        path = self.analytic_expansion(current, goal, ox, oy, kd_tree, ax)

        if path:
            if show_animation:
                ax.plot(path.x, path.y)
            f_x = path.x[1:]
            f_y = path.y[1:]
            f_yaw = path.yaw[1:]

            f_cost = current.cost + self.calc_rs_path_cost(path)
            f_parent_index = self.calc_index(current, c)

            fd = []
            f_vels = []
            f_steers = []
            for d in path.directions[1:]:
                fd.append(d >= 0)
                f_vels.append(d*self.Car.maxVel)

            for steer in path.steers[1:]:
                if steer=='L':                    
                    f_steers.append(self.Car.maxSteerAngle)
                elif steer=='R':                    
                    f_steers.append(-self.Car.maxSteerAngle)
                else:
                    f_steers.append(0.0)

            f_path = Node(current.x_index, current.y_index, current.yaw_index,
                        current.direction, f_x, f_y, f_yaw, fd, f_vels, f_steers,
                        cost=f_cost, parent_index=f_parent_index, steer=0.0)
            return True, f_path

        return False, None


    def calc_rs_path_cost(self, reed_shepp_path):
        cost = 0.0
        for length in reed_shepp_path.lengths:
            if length >= 0:  # forward
                cost += length
            else:  # back
                cost += abs(length) * BACK_COST

        # switch back penalty
        for i in range(len(reed_shepp_path.lengths) - 1):
            # switch back
            if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
                cost += SB_COST

        # steer penalty
        for course_type in reed_shepp_path.ctypes:
            if course_type != "S":  # curve
                cost += STEER_COST * abs(self.Car.maxSteerAngle)

        # ==steer change penalty
        # calc steer profile
        n_ctypes = len(reed_shepp_path.ctypes)
        u_list = [0.0] * n_ctypes
        for i in range(n_ctypes):
            if reed_shepp_path.ctypes[i] == "R":
                u_list[i] = - self.Car.maxSteerAngle
            elif reed_shepp_path.ctypes[i] == "L":
                u_list[i] = self.Car.maxSteerAngle

        for i in range(len(reed_shepp_path.ctypes) - 1):
            cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])

        return cost


    def hybrid_a_star_planning(self, start, goal, ox, oy, ax):
        """
        start: start node
        goal: goal node
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        """

        start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
        tox, toy = ox[:], oy[:]

        obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

        config = Config(tox, toy, self.xy_resolution, self.yaw_resolution)

        start_node = Node(round(start[0] / self.xy_resolution),
                        round(start[1] / self.xy_resolution),
                        round(start[2] / self.yaw_resolution), True,
                        [start[0]], [start[1]], [start[2]], [True], [0.0], [0.0], cost=0)
        goal_node = Node(round(goal[0] / self.xy_resolution),
                        round(goal[1] / self.xy_resolution),
                        round(goal[2] / self.yaw_resolution), True,
                        [goal[0]], [goal[1]], [goal[2]], [True], [0.0], [0.0])

        openList, closedList = {}, {}

        h_dp = calc_distance_heuristic(
            goal_node.x_list[-1], goal_node.y_list[-1],
            ox, oy, self.xy_resolution, self.Car.bubble_r)

        pq = []
        openList[self.calc_index(start_node, config)] = start_node
        heapq.heappush(pq, (self.calc_cost(start_node, h_dp, config),
                            self.calc_index(start_node, config)))
        final_path = None

        while True:
            if not openList:
                print("Error: Cannot find path, No open set")
                return [], False

            cost, c_id = heapq.heappop(pq)
            if c_id in openList:
                current = openList.pop(c_id)
                closedList[c_id] = current
            else:
                continue

            if show_animation:  # pragma: no cover
                ax.plot(current.x_list[-1], current.y_list[-1], "xc")
                ## for stopping simulation with the esc key.
                # ax.gcf().canvas.mpl_connect(
                #     'key_release_event',
                #     lambda event: [exit(0) if event.key == 'escape' else None])
                # if len(closedList.keys()) % 10 == 0:
                #     ax.pause(0.001)

            is_updated, final_path = self.update_node_with_analytic_expansion(
                current, goal_node, config, ox, oy, obstacle_kd_tree, ax)


            if is_updated:
                # print("path found")
                break
            
            # valid_neighbors = self.get_neighbors(current, config, ox, oy, obstacle_kd_tree)
            # for neighbor in valid_neighbors:
            for neighbor in self.get_neighbors(current, config, ox, oy,
                                        obstacle_kd_tree, ax):            
                neighbor_index = self.calc_index(neighbor, config)
                if neighbor_index in closedList:
                    continue
                if neighbor_index not in openList \
                        or openList[neighbor_index].cost > neighbor.cost:
                    heapq.heappush(
                        pq, (self.calc_cost(neighbor, h_dp, config),
                            neighbor_index))
                    openList[neighbor_index] = neighbor

        path = self.get_final_path(closedList, final_path)
        return path, True


    def calc_cost(self, n, h_dp, c):
        ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
        if ind not in h_dp:
            return n.cost + 999999999  # collision cost
        return n.cost + H_COST * h_dp[ind].cost

    def get_final_path(self, closed, goal_node):
        reversed_x, reversed_y, reversed_yaw = \
            list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
            list(reversed(goal_node.yaw_list))
        direction = list(reversed(goal_node.directions))
        vels = list(reversed(goal_node.vel_list))
        steers = list(reversed(goal_node.steer_list))

        nid = goal_node.parent_index
        final_cost = goal_node.cost

        while nid:
            n = closed[nid]
            reversed_x.extend(list(reversed(n.x_list)))
            reversed_y.extend(list(reversed(n.y_list)))
            reversed_yaw.extend(list(reversed(n.yaw_list)))
            direction.extend(list(reversed(n.directions)))
            vels.extend(list(reversed(n.vel_list)))
            steers.extend(list(reversed(n.steer_list)))

            nid = n.parent_index

        reversed_x = list(reversed(reversed_x))
        reversed_y = list(reversed(reversed_y))
        reversed_yaw = list(reversed(reversed_yaw))
        direction = list(reversed(direction))
        vels = list(reversed(vels))
        steers = list(reversed(steers))

        # adjust first and last control
        direction[0] = direction[1]
        vels[0] = vels[1]
        steers[0] = steers[1]

        direction[-1] = 0
        vels[-1] = 0.0
        steers[-1] = 0.0

        path = Path(reversed_x, reversed_y, reversed_yaw, direction, vels, steers, final_cost)

        return path


    def verify_index(self, node, c):
        x_ind, y_ind = node.x_index, node.y_index
        if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
            return True

        return False


    def calc_index(self, node, c):
        ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
            (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

        if ind <= 0:
            print("Error(calc_index):", ind)

        return ind

    def parallel_run(self, start, g_list, ox, oy, ax):
        # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        #     results = pool.starmap(self.hybrid_a_star_planning, [(start, goal, ox, oy, ax) for goal in g_list])
        results = [self.hybrid_a_star_planning(start, goal, ox, oy, ax) for goal in g_list]
        return results

def plot_anim(ax, fig, p_all, HA_obj):

    # Define the update function for the animation
    extend_end = 5
    time_traj = np.arange(p_all.shape[0])
    p_all = np.vstack((p_all, [p_all[-1]]*extend_end))
    time_traj = np.hstack((time_traj, [time_traj[-1]]*extend_end))

    end_time = int(time_traj[-1])

    car_plot, arrow_plot = HA_obj.Car.plot_car_return(p_all[0, 0], p_all[0, 1], p_all[0, 2], ax)

    time_text = 't=' + str(time_traj[0])
    props = dict(boxstyle='round', facecolor='w', alpha=0.5, edgecolor='black', linewidth=2)
    text_t  = ax.text(2.5,30.5, time_text, fontsize=22, bbox=props)

    def update(frame):

        p_yaw = p_all[frame, 2]
        p_x =  p_all[frame, 0]
        p_y =  p_all[frame, 1]

        rot = Rot.from_euler('z', -p_yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(HA_obj.Car.VRX, HA_obj.Car.VRY):
            converted_xy = np.stack([rx, ry]).T @ rot
            car_outline_x.append(converted_xy[0] + p_x)
            car_outline_y.append(converted_xy[1] + p_y)

        car_plot.set_data(car_outline_x, car_outline_y)
        arrow_plot.xy = [p_x + 1 * np.cos(p_yaw), p_y + 1 * np.sin(p_yaw)]
        arrow_plot.set_position((p_x, p_y))

        text_t.set_text('t=' + str(time_traj[frame]))

        # car = drawCar(Car_obj, x[frame], y[frame], yaw[frame])
        # car_plot_a.set_data(car[0, :], car[1, :])
        # arrow_plot_a.xy = [x[frame]+1*math.cos(yaw[frame]), y[frame]+1*math.sin(yaw[frame])]
        # arrow_plot_a.set_position((x[frame], y[frame]))
        # text_t.set_text('t=' + str(time_traj[frame]))
        # for i in range(len(dynamic_plot_a)):
        #     # Update circle positions
        #     dynamic_plot_a[i].center = (obst_x[frame][i], obst_y[frame][i],)
        # # dynamic_plot_a.set_data(obst_x[frame], obst_y[frame])

        plot_list = [car_plot] + [arrow_plot] + [text_t]
        return plot_list

    # Create the animation
    ani = FuncAnimation(fig, update, frames=p_all.shape[0], blit=True, interval=500, repeat_delay = 1000)

    save_file_name = 'Anim'
    file_dir_anim =  save_file_name + '.mp4'
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=-1)
    ani.save(file_dir_anim, writer=writer)

    plt.show()

def main():
    print("Start Hybrid A* planning")

    ## Get the parameters of the car and map from *.json files
    script_path = os.path.dirname(os.path.abspath(sys.argv[0])) 
    home_path = os.path.abspath(os.getcwd())
    config_path = home_path + '/Config/'
    type='big_lot'
    with open(config_path + 'config_planner.json') as f:
        config_planner = json.load(f)

    with open(config_path + 'config_map.json') as f:
        config_map = json.load(f)

    ## Initialize figures for plotting the trajectory and the animation
    fig, ax = plt.subplots(figsize=(10, 8))
    figa, axa = plt.subplots(figsize=(10, 8))
    axes = [ax, axa]

    ## Create the Hybrid A star planner object
    HA_obj = HA_planner(config_planner)

    ## create a map from the json file
    x_min, x_max, y_min, y_max, p_w, p_l, l_w, n_r, n_s, n_s1, obstacleX, obstacleY, s, center_spots, occ_spot_indices, roads_x, roads_y = map_lot(type, config_map, HA_obj.Car, axes)
    """
    ----Returns----
    x_min, x_max, y_min, y_max: 2D bounds of the parking lot area
    p_w: width of parking space
    p_l: length of parking space
    l_w: lane width
    n_r: number of parking rows, where each row has parking spaces on either side of the parking center line 
    n_s: number of parking spaces in each row, should be even
    n_s1: n_s/2
    obstacleX: list of x coordinates of static obstacles
    obstacleY: list of y coordinates of static obstacles
    s: start position of ego
    center_spots: (x, y) coordinates of centers of each parking spot
    occ_spot_indices: indices of parking spots that are occupied by static vehicles
    roads_x, roads_y: x and y coordinates of the center of the roads (all roads are two way)
    """

    ## Transform from center of vehicle to center of rear axle
    d = HA_obj.Car.wheelBase/2.0
    start = [s[0] - d*np.cos(s[2]), s[1] - d*np.sin(s[2]), s[2]]
    goal_yaw = 0.0*np.pi
    goal = [center_spots[35, 0]- d*np.cos(goal_yaw), center_spots[35, 1]- d*np.sin(goal_yaw), goal_yaw]
    print("start : ", start)
    print("goal : ", goal)
    
    time_all = []
    for _ in range(1):
        start_t = time.time()
        ## Running the planner, where input is <start, goal, obstacle locations, figure for plotting (ax)> 
        # Output is a path "result", that is a sequence of (x, y, theta, velocity, steering) with a fixed dt defined in config_planner.json""
        result = HA_obj.hybrid_a_star_planning(start, goal, obstacleX, obstacleY, ax)
        time_all.append(time.time()-start_t)

    print(f"Planner Computation time: Mean = {np.mean(time_all)}, STD={np.std(time_all)}")
    # print(f"Planner Computation time:", time_all)

    ## Plot the car's trajectory
    p_all = np.array([result[0].x_list, result[0].y_list, result[0].yaw_list]).T
    t = p_all.shape[0]
    for time_dynamic in range(t):

        HA_obj.Car.plot_car_trans(p_all[time_dynamic, 0], p_all[time_dynamic, 1], p_all[time_dynamic, 2], ax)

    for ax in axes:
        ax.axis('equal')

    plot_anim(axa, fig, p_all, HA_obj)
    plt.show()

if __name__ == '__main__':
    main()