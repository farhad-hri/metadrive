"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import sys
import pathlib
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from math import cos, sin, tan, pi

import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d

## Getting car dimension from config_planner.json file
import json
import os

script_path = os.path.dirname(os.path.abspath(sys.argv[0])) 

## Load the configuration files
config_path = script_path + '/Config/'

with open(config_path + 'config_planner.json') as f:
    config_planner = json.load(f)

WB = config_planner['vehicle-params']['wheelbase_length']  # rear to front wheel
W = config_planner['vehicle-params']['vehicle_width']  # width of car
# assuming front axle to front end is same as rear axle to back end
length = WB + 2*config_planner['vehicle-params']['axle_to_back']
LF = WB + config_planner['vehicle-params']['axle_to_back']  # distance from rear to vehicle front end
LB = config_planner['vehicle-params']['axle_to_back']  # distance from rear to vehicle back end
MAX_STEER = config_planner['vehicle-params']['max_steer']  # [rad] maximum steering angle

BUBBLE_DIST = (LF - LB) / 2.0  # distance from rear to center of vehicle.
BUBBLE_R = np.hypot((LF + LB) / 2.0, W / 2.0)  # bubble radius

SAFE_MARGIN = config_planner['HA*-params']['safety_margin']
# vehicle rectangle vertices
VRX = [LF, LF, -LB, -LB, LF]
VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]


def check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
        cx = i_x + BUBBLE_DIST * cos(i_yaw)
        cy = i_y + BUBBLE_DIST * sin(i_yaw)

        ids = kd_tree.query_ball_point([cx, cy], BUBBLE_R)

        if not ids:
            continue

        if not rectangle_check(i_x, i_y, i_yaw,
                               [ox[i] for i in ids], [oy[i] for i in ids]):
            return False  # collision

    return True  # no collision


def rectangle_check(x, y, yaw, ox, oy):
    # transform obstacles to base link frame
    rot = rot_mat_2d(yaw)
    for iox, ioy in zip(ox, oy):
        tx = iox - x
        ty = ioy - y
        converted_xy = np.stack([tx, ty]).T @ rot
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > (LF+SAFE_MARGIN) or rx < (-LB-SAFE_MARGIN) or ry > (W / 2.0+SAFE_MARGIN) or ry < (-W / 2.0-SAFE_MARGIN)):
            return False  # collision

    return True  # no collision


def plot_arrow(x, y, yaw, ax, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw, ax)
    else:
        ax.arrow(x, y, length * cos(yaw), length * sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)


def plot_car(x, y, yaw, ax):
    car_color = '-k'
    c, s = cos(yaw), sin(yaw)
    rot = rot_mat_2d(-yaw)
    car_outline_x, car_outline_y = [], []
    for rx, ry in zip(VRX, VRY):
        converted_xy = np.stack([rx, ry]).T @ rot
        car_outline_x.append(converted_xy[0]+x)
        car_outline_y.append(converted_xy[1]+y)

    arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
    plot_arrow(arrow_x, arrow_y, arrow_yaw, ax)

    ax.plot(car_outline_x, car_outline_y, car_color)


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw = pi_2_pi(yaw + distance * tan(steer) / L)  # distance/2

    return x, y, yaw


# def main():
#     x, y, yaw = 0., 0., 1.
#     plt.axis('equal')
#     plot_car(x, y, yaw)
#     plt.show()


# if __name__ == '__main__':
#     main()