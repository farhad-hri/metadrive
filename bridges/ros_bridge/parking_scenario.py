# Import necessary classes and modules.
# MultiAgentParkingLotEnv: Environment simulating a parking lot with multiple agents.
# Pedestrian: A traffic participant that can be spawned in the environment.
from metadrive import MultiAgentParkingLotEnv
from metadrive.component.traffic_participants.pedestrian import Pedestrian
import pprint  # For pretty-printing data structures.
import numpy as np
import matplotlib.pyplot as plt

# Create an instance of the MultiAgentParkingLotEnv with custom configuration.
env = MultiAgentParkingLotEnv(
    {
        "horizon": 100000,  # Maximum number of simulation steps per episode.
        "vehicle_config": {
            "lidar": {
                "num_lasers": 72,  # Number of laser beams for the lidar sensor.
                "num_others": 0,   # Additional lidar configurations (none in this case).
                "distance": 40     # Maximum distance the lidar can sense.
            },
            "show_lidar": False,   # Disable the display of lidar visuals.
        },
        "debug_static_world": False,  # Disable static world debugging.
        "debug_physics_world": False, # Disable physics world debugging.
        "use_render": True,          # Do not render the simulation visually.
        "debug": True,                # Enable debug mode for more verbose logging.
        "manual_control": True,       # Allow manual control of agents for debugging.
        "num_agents": 2,              # Set the number of agents (vehicles) in the simulation.
        "delay_done": 10,             # Delay before an episode is marked as done.
        "parking_space_num": 10,       # Number of parking spaces available in the lot.
    }
)

# Reset the environment to start a new simulation episode.
# The reset() method returns initial observations and additional information.
o, _ = env.reset()

# Spawn a pedestrian object in the simulation.
# - Position: [30, 0] (x, y coordinates).
# - heading_theta: 0 radians (orientation).
# - random_seed: Ensures reproducible behavior for this object.
obj_1 = env.engine.spawn_object(Pedestrian, position=[30, 0], heading_theta=0, random_seed=1)

# Set the pedestrian's velocity.
# - Velocity vector: [1, 0] in the local frame.
# - Speed: 1 unit.
obj_1.set_velocity([1, 0], 1, in_local_frame=True)

# Initialize accumulators (or placeholders) for total reward and episode start time.
total_r = 0
ep_s = 0

# Retrieve the current map from the environment which holds the layout and other details.
current_map = env.current_map

# Access the list of parking spots from the current map.
parking_spots = current_map.parking_space

# Pretty-print the entire current_map for debugging and inspection.
pprint.pprint(current_map)

# Retrieve dimensions of the parking lot's parking spaces.
parking_length = current_map.parking_lot.parking_space_length
parking_width = current_map.parking_lot.parking_space_width

# Print out the attributes and methods available in the parking_spots object.
print(dir(parking_spots))

# Print the total number of parking spots available.
print("Total parking spots available:", len(parking_spots))

fig, ax = plt.subplots(figsize=(10, 8))

parking_lanes_lines = []
parking_lanes = []
center_spots = []

# Loop through each parking spot and display detailed information.
for idx, spot in enumerate(parking_spots):
    # # Print available attributes and methods for the current parking spot.
    # print(dir(spot))
    
    # # Display key details about this parking spot.
    # print("Parking Spot {}:".format(idx))
    # print("  Start Node:", spot.start_node)  # Starting node of the spot.
    # print("  End Node:", spot.end_node)      # Ending node of the spot.
    
    # # Retrieve and potentially process lane information for this parking spot.
    # parking_lane = spot.get_lanes(current_map.road_network)
    # # To get center line for each spot
    # center_line = parking_lane[0].get_polyline()

    # # Optionally retrieve all lane information from the parking lot's block network.
    # all_lanes = current_map.parking_lot.block_network.get_all_lanes()
    # parking_lane[0].width_at(0.0) # provide longitudinal value
    # # Print a separator line for clarity between parking spot details.
    # print("-" * 30)

    lanes = spot.get_lanes(current_map.road_network)
    parking_lanes.append(lanes)
    lines = lanes[0].get_polyline()
    parking_lanes_lines.append(lines)
    center_spots.append([lines[0, 0], min(lines[:, 1]) + (max(lines[:, 1]) - min(lines[:, 1]))/2.0])
    # ax.plot(lines[:, 0], lines[:, 1], marker='o', color='black')
    ax.plot(center_spots[-1][0], center_spots[-1][1], marker='o', color='grey')    
    ax.plot(lines[:, 0] - parking_width/2.0, lines[:, 1], color='grey') 
    ax.plot(lines[:, 0] + parking_width/2.0, lines[:, 1], color='grey')

# Finally, pretty-print the entire parking_spots structure for a full overview.
# pprint.pprint(parking_spots)

other_agent_state = env.agents['agent1'].get_state()
other_agent_state['position'][0], other_agent_state['position'][1] = center_spots[0][0], center_spots[0][1]
env.agents['agent1'].set_state(other_agent_state)
 
ego_state = env.agents['agent0'].get_state()
ego_state['position'][0], ego_state['position'][1] = 15.0, 0.0
ego_state['heading_theta'] = 0.0
env.agents['agent0'].set_state(ego_state)

ax.plot(ego_state['position'][0], ego_state['position'][1], color='green', marker='o', markersize=12)

other_state = env.agents['agent1'].get_state()
ax.plot(other_state['position'][0], other_state['position'][1], color='red', marker='o', markersize=24)

ped = obj_1.get_state()
ax.plot(ped['position'][0], ped['position'][1], color='orange', marker='o', markersize=24)

for i in range(1, 1000000):
    actions = {k: [1.0, .0] for k in env.agents.keys()}
    if len(env.agents) == 1:
        actions = {k: [-1.0, .0] for k in env.agents.keys()}
    o, r, tm, tc, info = env.step(actions)
    obj_1.set_velocity([1, 0], 2, in_local_frame=True)
    for r_ in r.values():
        total_r += r_
    ep_s += 1
    # d.update({"total_r": total_r, "episode length": ep_s})
    if len(env.agents) != 0:
        v = env.current_track_agent
        dist = v.dist_to_left_side, v.dist_to_right_side
        ckpt_idx = v.navigation._target_checkpoints_index
    else:
        dist = (0, 0)
        ckpt_idx = (0, 0)

    render_text = {
        "total_r": total_r,
        "episode length": ep_s,
        "cam_x": env.main_camera.camera_x,
        "cam_y": env.main_camera.camera_y,
        "cam_z": env.main_camera.top_down_camera_height,
        "alive": len(env.agents),
        "dist_right_left": dist,
        "ckpt_idx": ckpt_idx,
        "parking_space_num": len(env.engine.spawn_manager.parking_space_available)
    }
    if len(env.agents) > 0:
        v = env.current_track_agent
        # print(v.navigation.checkpoints)
        render_text["current_road"] = v.navigation.current_road

    env.render(mode="topdown", semantic_map=True, text=render_text)
    d = tm
    for kkk, ddd in d.items():
        if ddd and kkk != "__all__":
            print(
                "{} done! State: {}".format(
                    kkk, {
                        "arrive_dest": info[kkk]["arrive_dest"],
                        "out_of_road": info[kkk]["out_of_road"],
                        "crash": info[kkk]["crash"],
                        "max_step": info[kkk]["max_step"],
                    }
                )
            )
    if tm["__all__"]:
        print(
            "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                i, total_r, total_r / env.agent_manager.next_agent_count
            )
        )
        env.reset()
        # break
    if len(env.agents) == 0:
        total_r = 0
        print("Reset")
        env.reset()

    current_position = np.array(env.agents['agent0'].get_state()['position'])[:2]
    current_heading = np.array(env.agents['agent0'].get_state()['heading_theta']) 
    v = 1.0
    if i < 10:
        acc = 1.0
    else:
        acc = 0.0
    m = env.agents['agent0'].MASS
    dt = 0.5
    max_steer = env.agents['agent0'].max_steering # 40.0 deg
    max_acc_f = env.agents['agent0'].config["max_engine_force"] 
    max_dacc_f = env.agents['agent0'].config["max_brake_force"] 
    steering = -10.0
    if i > 100000:
        steering = 0.0   
    steer_norm = steering/max_steer
    hp = np.abs((acc * v * m)/745.7)
    if acc >= 0.0:
        acc_norm = hp/max_acc_f
    else:
        acc_norm = -hp/max_dacc_f
    action = [0.0, 1.0]
    # action[0] = steer_norm
    # action[1] = acc_norm

    vel_global = np.array([v*np.cos(current_heading), v*np.sin(current_heading)]) 
    next_position = current_position[:2] +  dt*vel_global
    # heading = current_heading + (v/2.0)*np.tan(steering)*dt
    # env.agents['agent0'].set_position(next_position.tolist())
    # ego_state = env.agents['agent0'].get_state()
    # ego_state['position'][0], ego_state['position'][1] = next_position[0], next_position[1]
    # ego_state['velocity'] = vel_global
    # ego_state['heading_theta'] = 0.0
    # env.agents['agent0'].set_state(ego_state) 
    env.agents['agent0'].set_velocity(vel_global)
    # env.agents['agent0'].set_heading_theta(heading)
    # env.agents['agent0'].set_steering(steering)
    # env.agents['agent0']._apply_throttle_brake(1.0)
    print(f"Ego acceleration = {env.agents['agent0'].get_state()['throttle_brake']}, steering = {env.agents['agent0'].get_state()['steering']}")
    # print("Ego action: ", env.agents['agent0'].get_state()['throttle_brake'])
env.close()