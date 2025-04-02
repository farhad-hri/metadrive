from metadrive import (
 MultiAgentParkingLotEnv
)
from metadrive.component.traffic_participants.pedestrian import Pedestrian
import pprint
import matplotlib.pyplot as plt

env = MultiAgentParkingLotEnv(
    {
        "horizon": 100000,
        "vehicle_config": {
            "lidar": {
                "num_lasers": 72,
                "num_others": 0,
                "distance": 40
            },
            "show_lidar": True,
        },
        "debug_static_world": False,
        "debug_physics_world": False,
        "use_render": True,
        "debug": False,
        "manual_control": True,
        "num_agents": 2,
        "delay_done": 10,
        "parking_space_num": 10,
    }
)
o, _ = env.reset()
obj_1 = env.engine.spawn_object(Pedestrian, position=[30, 0], heading_theta=0, random_seed=1)
obj_1.set_velocity([1, 0], 1, in_local_frame=True)
total_r = 0
ep_s = 0

current_map = env.current_map
# List all parking spots.
# Depending on your map implementation, parking_space might be a list or dict of destination roads.
parking_spots = current_map.parking_space
parking_length = current_map.parking_lot.parking_space_length
parking_width = current_map.parking_lot.parking_space_width
print("Total parking spots available:", len(parking_spots))


fig, ax = plt.subplots(figsize=(10, 8))

parking_lanes_lines = []
parking_lanes = []
center_spots = []
for idx, spot in enumerate(parking_spots):
    lanes = spot.get_lanes(current_map.road_network)
    parking_lanes.append(lanes)
    lines = lanes[0].get_polyline()
    parking_lanes_lines.append(lines)
    center_spots.append([lines[0, 0], min(lines[:, 1]) + (max(lines[:, 1]) - min(lines[:, 1]))/2.0])
    # ax.plot(lines[:, 0], lines[:, 1], marker='o', color='black')
    ax.plot(center_spots[-1][0], center_spots[-1][1], marker='o', color='grey')    
    ax.plot(lines[:, 0] - parking_width/2.0, lines[:, 1], color='grey') 
    ax.plot(lines[:, 0] + parking_width/2.0, lines[:, 1], color='grey')

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

for i in range(1, 1000):
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


# ax.plot(lines[:, 0] + parking_width/2.0, lines[:, 1], color='grey')
ax.axis('equal')
plt.show()