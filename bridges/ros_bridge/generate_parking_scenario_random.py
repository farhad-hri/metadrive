from metadrive import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.map.pg_map import MapGenerateMethod
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
import logging

map_config={BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE, 
            BaseMap.GENERATE_CONFIG: "XOS",  # 3 block
            BaseMap.LANE_WIDTH: 3.5,
            BaseMap.LANE_NUM: 1}

fig, axs = plt.subplots(1, 10, figsize=(10, 10), dpi=200)
plt.tight_layout(pad=-3)

for i in range(1):
    #if i==0:
    map_config["config"]="PTTP"
    map_config["lane_num"]=1
    env = MetaDriveEnv(dict(num_scenarios=10, map_config=map_config, log_level=logging.WARNING))
    # elif i==1:
    #     map_config["config"]="PT"
    #     map_config["lane_num"]=1
    #     # map_config["parking_space_num"] = 2
    #     env = MetaDriveEnv(dict(num_scenarios=10, map_config=map_config, log_level=logging.WARNING))
    #     # env.current_map.
    # elif i==2:
    #     map_config["config"]="PTTP"
    #     map_config["lane_num"]=1
    #     env = MetaDriveEnv(dict(num_scenarios=10, map_config=map_config, log_level=logging.CRITICAL))
    for j in range(10):
        env.reset(seed=j)
        m = draw_top_down_map(env.current_map)
        ax = axs[j]
        ax.imshow(m, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
    env.close()
plt.show()