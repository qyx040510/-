import serial
import time
import threading
import re
import numpy as np
from scipy.ndimage import convolve
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import sys
from skimage.measure import label, regionprops
import csv
from datetime import datetime

GLOBAL_PORT = "/dev/tty.usbserial-54FC0361651"
array_2d = [[0 for _ in range(32)] for _ in range(32)]
xx_index = {}
data_lock = threading.Lock()
value_threshold = 4500  # 需要根据具体情况调整
density_threshold = 5  # 需要根据具体情况调整
units = []  # 存储符合条件的步态数据
gait_cycle = False  # 标记是否在步态周期内
array_32 = []  # 用于存储32行数据

# 假设预测模型返回一个置信度和预测结果
def predict_user(features):
    # 这里使用虚拟数据来模拟预测结果和置信度
    confidence = np.random.random()
    if confidence > 0.8:
        return 0, confidence  # 员工A
    elif confidence > 0.6:
        return 1, confidence  # 员工B
    elif confidence > 0.4:
        return 2, confidence  # 员工C
    else:
        return -1, confidence  # 未登记用户

def read_from_serial(port=GLOBAL_PORT, baudrate=2000000):
    global gait_cycle, array_32  # 使用全局变量
    ser = serial.Serial(port, baudrate, timeout=5)
    print(f"Listening on {port} at {baudrate} baudrate...")

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                match = re.search(r'(\d+)PressureArray:(.*),', line)
                if match is not None:
                    xx_value = int(match.group(1))
                    c_values_str = match.group(2)
                    with data_lock:
                        xx_index[xx_value] = c_values_str
                        for key, value in xx_index.items():
                            row = key
                            values = value.split(",")
                            for i, v in enumerate(values):
                                if i < 32:
                                    try:
                                        array_2d[row][i] = (int(v)*2)
                                    except ValueError:
                                        print(f"Failed to convert '{v}' to int for row {row}, column {i}")
                                else:
                                    print(f"Column index {i} is out of range for row {row}")

                        if xx_value == 31:  # 每收集到32行数据处理一次
                            array_32 = np.copy(array_2d)
                            process_gait_cycle(array_32)

    except KeyboardInterrupt:
        print("Stopped listening.")
        sys.exit()
    finally:
        ser.close()

def process_gait_cycle(data):
    global gait_cycle
    # 计算步态周期的开始和结束
    binary_unit = (data > value_threshold).astype(int)
    kernel = np.ones((3, 3))
    density = convolve(binary_unit, kernel, mode='constant', cval=0.0)
    max_density = np.max(density)

    if max_density > density_threshold:
        if not gait_cycle:
            gait_cycle = True
            print("Gait cycle detected.")
        units.append(np.copy(data))
        print(f"Units length during gait cycle: {len(units)}")
    elif gait_cycle and max_density <= density_threshold:
        print("Gait cycle ending...")

        centroids = []
        for unit in units:
            # 标记连通区域
            labeled_data, num_features = label(unit, return_num=True)
            # 提取脚掌区域（假设最大的连通区域为脚掌）
            regions = regionprops(labeled_data)
            if regions:
                largest_region = max(regions, key=lambda region: region.area)
                centroid = largest_region.centroid
            # 预处理，提取脚掌区域和质心
            centroids.append(centroid)
        direction, direction_vector = compute_overall_direction(centroids)
        print(f"Direction of this gait cycle : {direction}")

        features = compute_and_save_features(units)
        predict_and_save_results(features, direction)
        units.clear()
        gait_cycle = False
        print("Gait cycle ended and saved.")

def compute_and_save_features(units):
    all_features = []
    for idx, unit in enumerate(units):
        features = compute_force_features(unit)
        save_features_to_file(features, idx)
        all_features.append(features)
    return all_features

def compute_force_features(cleaned_data):
    features = {}

    # 统计特征
    features['mean'] = np.mean(cleaned_data)
    features['min'] = np.min(cleaned_data)
    feature_max = np.max(cleaned_data)
    features['range'] = feature_max - features['min']
    features['skewness'] = np.mean(np.power(cleaned_data - features['mean'], 3)) / np.power(features['range'], 3/2)
    features['kurtosis'] = np.mean(np.power(cleaned_data - features['mean'], 4)) / np.power(features['range'], 2) - 3
    features['gait_duration'] = cleaned_data.shape[0]

    # 空间特征
    labeled_data, num_features = label(cleaned_data > 0, return_num=True)
    regions = regionprops(labeled_data)
    if regions:
        largest_region = max(regions, key=lambda region: region.area)
        features['area'] = largest_region.area
        features['centroid_y'] = largest_region.centroid[1]
        features['eccentricity'] = largest_region.eccentricity
        features['orientation'] = largest_region.orientation

    return features

# 计算行走方向
def compute_overall_direction(centroids):
    if len(centroids) < 2:
        return 'unknown', np.array([0, 0, 0])  # 返回默认值

    start = centroids[0]
    end = centroids[-1]

    dx = end[1] - start[1]
    dy = end[0] - start[0]

    angle = np.arctan2(dy, dx) * 180 / np.pi

    print(angle)
    if 0 < angle <= 180:
        return 'in', np.array([0, -1, 0])
    elif -180 < angle <= 0:
        return 'out', np.array([0, 1, 0])
    else:
        return 'unknown', np.array([0, 0, 0])

def predict_and_save_results(features, direction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    combined_features = np.mean([list(feature.values()) for feature in features], axis=0)
    user_id, confidence = predict_user(combined_features)

    if user_id == 0:
        user_label = '员工A'
    elif user_id == 1:
        user_label = '员工B'
    elif user_id == 2:
        user_label = '员工C'
    else:
        user_label = '未登记用户'

    with open('results.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([timestamp, user_label, confidence, direction])

def save_gait_cycle_to_file(units):
    timestamp = int(time.time() * 1000)
    filename = f'gait_cycle_{timestamp}.txt'
    print(f"Saving data to {filename} with {len(units)} units")
    try:
        with open(filename, 'w') as file:
            print("File opened for writing")
            for unit in units:
                for row in unit:
                    file.write(','.join(map(str, row)) + '\n')
                file.write('\n')  # 每个步态数据之间加一个空行以便区分
            print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Exception occurred while saving data: {e}")

def save_features_to_file(features, idx):
    timestamp = int(time.time() * 1000)
    filename = f'features_{timestamp}_{idx}.txt'
    print(f"Saving features to {filename}")
    try:
        with open(filename, 'w') as file:
            for key, value in features.items():
                file.write(f"{key}: {value}\n")
            print(f"Features saved to {filename}")
    except Exception as e:
        print(f"Exception occurred while saving features: {e}")

def save_time_interval_to_file(time_interval, filename):
    with open(filename, 'a') as file:
        file.write(f'{time_interval}\n')

def display_grid_dynamic(data_func):
    cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', [(0, 'red'), (0.5, 'white'), (1, 'blue')]
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(data_func(), cmap='viridis_r', vmin=4000, vmax=5000)
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', ticks=[4000, 5000], format='%d')

    plt.ion()
    count = 0
    last_time = time.time()

    try:
        while True:
            data = data_func()
            cax.set_data(data)
            ax.figure.canvas.draw_idle()
            if count % 1 == 0:
                current_time = time.time()
                time_interval = current_time - last_time
                last_time = current_time
                save_time_interval_to_file(time_interval, 'time_intervals.txt')
            count += 1
            plt.pause(0.1)
    except KeyboardInterrupt:
        print("Display terminated.")
        sys.exit()
    finally:
        plt.ioff()


def changenumber_hang():
    for i in range(0, len(array_2d), 2):
        for j in range(32):
            temp = array_2d[i][j]
            array_2d[i][j] = array_2d[i + 1][j]
            array_2d[i + 1][j] = temp


def changenumber_lie():
    for row in array_2d:
        for i in range(0, len(row), 2):
            temp = row[i]
            row[i] = row[i + 1]
            row[i + 1] = temp


def external_data_func():
    return array_2d


if __name__ == '__main__':
    # 创建线程对象
    thread1 = threading.Thread(target=read_from_serial, args=(GLOBAL_PORT, 2000000))
    # 启动串口读取线程
    thread1.start()
    # 在主线程中显示图形
    display_grid_dynamic(external_data_func)
    # 等待串口读取线程结束
    thread1.join()
    print("Both threads have finished.")