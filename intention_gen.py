import carla
import random
import time
import math
import pickle
import numpy as np
from datetime import datetime
from collections import deque

# 连接到CARLA服务器
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
map_name = 'Town05'
world = client.load_world(map_name)
world_map = world.get_map()
waypoints = world_map.generate_waypoints(2.0)

traffic_lights = world.get_actors().filter('traffic.traffic_light')

import pandas as pd
data_list = []

for waypoint in world_map.generate_waypoints(1.0):
    x = waypoint.transform.location.x
    y = waypoint.transform.location.y
    road_id = waypoint.road_id
    lane_id = waypoint.lane_id
    s = waypoint.s
    data_list.append([x, y, road_id, lane_id, s])

map_df = pd.DataFrame(data_list, columns=['x', 'y', 'road_id', 'lane_id', 's'])


import math
import pickle

# 获取十字路口
junctions = [waypoint for waypoint in waypoints if waypoint.is_junction]

# 分组每个路口的路点
junctions_and_waypoints = {}
for waypoint in waypoints:
    for junction in junctions:
        if waypoint.is_junction and waypoint.junction_id == junction.junction_id:
            if junction not in junctions_and_waypoints:
                junctions_and_waypoints[junction] = []
            junctions_and_waypoints[junction].append(waypoint)

for junction, waypoints in junctions_and_waypoints.items():
    # 绘制十字路口前50米的路点
    previous_waypoint = junction.previous(50)[0]
    world.debug.draw_string(previous_waypoint.transform.location, 'Pre', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=100)
    

    # 获取十字路口的边界框
    junction_bounding_box = junction.get_junction().bounding_box

    # 获取车道
    lane_id, road_id = previous_waypoint.lane_id, previous_waypoint.road_id
    road_data = map_df[(map_df['lane_id'] == lane_id) & (map_df['road_id'] == road_id)]
    first_row = road_data.iloc[0]
    # world.debug.draw_point(carla.Location(x=first_row['x'], y=first_row['y']), size=0.1, color=carla.Color(255, 0, 0), life_time=100)
    stop_line_y = first_row['y']

    lane_width = previous_waypoint.lane_width
    left = carla.Location(x=first_row['x'] + lane_width * 1.5, y=stop_line_y)
    right = carla.Location(x=first_row['x'] - lane_width * 0.5, y=stop_line_y)
    world.debug.draw_string(left, 'Left', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=100)
    world.debug.draw_string(right, 'Right', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=100)


    # break
    
    trajectory_data = []
    num = 0
    # vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle.audi.*'))
    vehicle_bp = world.get_blueprint_library().find("vehicle.audi.a2")
    print(vehicle_bp.id)
    for i in range(50):
        print(i, num)
        vehicle = world.try_spawn_actor(vehicle_bp, previous_waypoint.transform)
        
        if vehicle is not None:
            print(f'Created {vehicle_bp.id}')
            num += 1
            
            # # 设置车头方向
            # direction_vector = previous_waypoint.next(2.0)[0].transform.location - previous_waypoint.transform.location
            # direction_vector = direction_vector / direction_vector.length()
            # yaw = math.atan2(direction_vector.y, direction_vector.x) * (180.0 / math.pi)
            # new_transform = carla.Transform(location=vehicle.get_location(), rotation=carla.Rotation(yaw=yaw))
            # vehicle.set_transform(new_transform)

            # 启用车辆自动驾驶
            vehicle.set_autopilot(True)

            
            # 设置目标路口的红绿灯为绿色
            for traffic_light in traffic_lights:
                traffic_light.set_state(carla.TrafficLightState.Green)

            # 记录车辆轨迹，持续10秒
            trajectory = []
            for _ in range(100):
                # 获取车辆的位置和朝向
                vehicle_location = vehicle.get_location()
                vehicle_rotation = vehicle.get_transform().rotation
                trajectory.append([vehicle_location.x, vehicle_location.y, vehicle_rotation.yaw])
                # world.debug.draw_point(vehicle_location, size=0.1, color=carla.Color(0, 255, 0), life_time=1000)
                time.sleep(0.1)

            trajectory_data.append(trajectory[:])






       

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    pickle_filename = f"./traj/vehicle_trajectory_{current_time}.pkl"
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(trajectory_data, pickle_file)

    break




