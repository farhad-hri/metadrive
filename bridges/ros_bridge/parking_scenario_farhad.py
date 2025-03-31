from metadrive import (
 MultiAgentParkingLotEnv
)
from metadrive.component.traffic_participants.pedestrian import Pedestrian
import pprint

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
        "parking_space_num": 4,
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
pprint.pprint(current_map)
parking_length = current_map.parking_lot.parking_space_length
parking_width = current_map.parking_lot.parking_space_width
print(dir(parking_spots))
# For clarity, we print out each parking spot's key details.
# In this example, we assume each parking spot is represented as a Road object.
print("Total parking spots available:", len(parking_spots))
for idx, spot in enumerate(parking_spots):
    print(dir(spot))
    # Display some representative info about the spot (e.g., start and end nodes)
    print("Parking Spot {}:".format(idx))
    print("  Start Node:", spot.start_node)
    print("  End Node:", spot.end_node)
    print(current_map.road_network.shortest_path(spot.start_node, spot.end_node))
    print("  Length:", spot.length if hasattr(spot, "length") else "N/A")
    pprint.pprint(spot.lane_index)
    # For spot lane 
    spot.get_lanes(current_map.road_network)
    # For all lanes
    current_map.parking_lot.block_network.get_all_lanes()
    #print(current_map.road_network.get_lane(spot.lane_index))
    #print(current_map.parking_lot.block_network.get_all_lanes())
    print("-" * 30)

# Optionally, pretty-print the entire parking_space structure.
pprint.pprint(parking_spots)

pprint.pprint('Debug')