import carla
import random
import time
import math
import pickle
import numpy as np
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.autograd import Variable



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

# world.debug.draw_string(carla.Location(x=29.8315, y=76.7277), 'Junction', draw_shadow=False,
#                                 color=carla.Color(r=255, g=0, b=0), life_time=100)

# 局部坐标
# egox, egoy, egophi = 29.8315, 76.7277, - np.pi/2
# E_routeX = (G_routeX[nearest_idx:end_idx]-egox)*np.cos(egophi)+(G_routeY[nearest_idx:end_idx]-egoy)*np.sin(egophi) 
# E_routeY = -(G_routeX[nearest_idx:end_idx]-egox)*np.sin(egophi)+(G_routeY[nearest_idx:end_idx]-egoy)*np.cos(egophi) 

# 转换全局坐标到局部坐标
def global_to_local(global_traj, reference_point=(29.8, 78, np.pi/2)):
    local_traj = deque(maxlen=10)
    ref_x, ref_y, ref_yaw = reference_point

    for point in global_traj:
        x, y, yaw, velocity = point
        dx = x - ref_x
        dy = y - ref_y
        local_x = dx * np.cos(-ref_yaw) - dy * np.sin(-ref_yaw)
        local_y = dx * np.sin(-ref_yaw) + dy * np.cos(-ref_yaw)
        local_traj.append([local_x, local_y, yaw, velocity])

    # plt.plot([p[0] for p in local_traj], [p[1] for p in local_traj])
    # plt.scatter(local_traj[0][0], local_traj[0][1], marker='.', color='r', linewidth=0.5, alpha=0.5)
    # plt.axis('equal')
    # plt.show()
    # 将 NumPy 数组转换为 PyTorch 张量
    local_traj_tensor = torch.tensor(list(local_traj))
    return local_traj_tensor


torch.cuda.set_device(0)
device = 'cuda'
batch_size = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = x.view(-1, 40)
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x = x.view(batch_size, sequence_length, 1)
        
        # Apply LSTM
        h0 = torch.zeros(2, batch_size, 64).to(device)  # Initial hidden state
        c0 = torch.zeros(2, batch_size, 64).to(device)  # Initial cell state
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Reshape and pass through linear layer
        lstm_out = lstm_out[:, -1, :]  # Take the last time step output
        x = self.relu(lstm_out)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        
        return x

model = Net().to(device)
weights_path = 'model_weights.pth'
model.load_state_dict(torch.load(weights_path))
model.eval()

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
    stop_line_y = first_row['y']

    lane_width = previous_waypoint.lane_width
    left = carla.Location(x=first_row['x'] + lane_width * 1.5, y=stop_line_y)
    right = carla.Location(x=first_row['x'] - lane_width * 0.5, y=stop_line_y)


    vehicle_bp = world.get_blueprint_library().find("vehicle.audi.a2")
    print(vehicle_bp.id)

    total, correct, acc = 0, 0, 0

    for i in range(50):
        vehicle = world.try_spawn_actor(vehicle_bp, previous_waypoint.transform)
        
        if vehicle is not None:
            print(f'Created {vehicle_bp.id}')
            # 启用车辆自动驾驶
            vehicle.set_autopilot(True)

            # 设置目标路口的红绿灯为绿色
            # for traffic_light in traffic_lights:
            #     traffic_light.set_state(carla.TrafficLightState.Green)

            # 记录车辆轨迹
            traj = deque(maxlen=10)
            for _ in range(100):
                for traffic_light in traffic_lights:
                    traffic_light.set_state(carla.TrafficLightState.Green)
                # 获取车辆的位置和朝向
                vehicle_location = vehicle.get_location()
                if vehicle_location.y > 78:
                    break
                yaw = vehicle.get_transform().rotation.yaw
                velocity = vehicle.get_velocity().length()
                traj.append([vehicle_location.x, vehicle_location.y, yaw, velocity])
                time.sleep(0.1)
        
            local_traj = global_to_local(traj).to(device).float()
            outputs = model(local_traj).squeeze()
            _, pred_label = torch.max(outputs, 0)
            pred_label = pred_label.cpu().numpy()
            
            if pred_label == 0:
                world.debug.draw_string(carla.Location(x=vehicle.get_location().x, y=76.7277), 'Go Straight', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=1.5)
            elif pred_label == 1:
                world.debug.draw_string(carla.Location(x=vehicle.get_location().x, y=76.7277), 'Turn Right', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=1.5)
            elif pred_label == 2:
                world.debug.draw_string(carla.Location(x=vehicle.get_location().x, y=76.7277), 'Turn Left', draw_shadow=False,
                                color=carla.Color(r=255, g=0, b=0), life_time=1.5)
            
            for traffic_light in traffic_lights:
                traffic_light.set_state(carla.TrafficLightState.Green)

            # time.sleep(5)
            # total += 1
            
            # if vehicle.get_location().x < right.x and pred_label == 2:
            #     correct += 1
            # elif vehicle.get_location().x > left.x and pred_label == 1:
            #     correct += 1
            # elif vehicle.get_location().x < left.x and vehicle.get_location().x > right.x and pred_label == 0:
            #     correct += 1

            # # 显示 'total'
            # world.debug.draw_string(carla.Location(x=30, y=84), 'Total: {}'.format(total), draw_shadow=False,
            #                         color=carla.Color(r=255, g=0, b=0), life_time=5)

            # # 显示 'correct'
            # world.debug.draw_string(carla.Location(x=30, y=87), 'Correct: {}'.format(correct), draw_shadow=False,
            #                         color=carla.Color(r=255, g=0, b=0), life_time=5)

            # # 显示 'accuracy'
            # world.debug.draw_string(carla.Location(x=30, y=90), 'Accuracy: {:.2f}%'.format(correct / total * 100), draw_shadow=False,
            #                         color=carla.Color(r=255, g=0, b=0), life_time=5)
            
            # time.sleep(10)
    break




