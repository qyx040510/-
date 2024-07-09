import requests
import json
from datetime import datetime


def save_results_to_server(timestamp, prediction, direction):
    url = 'http://<server_ip>:5000/receive_data'  # 替换为服务器的IP地址
    data = {
        'timestamp': timestamp,
        'prediction': prediction,
        'direction': direction
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    if response.status_code == 200:
        print("Data successfully sent to server")
    else:
        print("Failed to send data to server")


def save_results_to_file_and_server(results):
    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results)

    timestamp, prediction, direction = results
    save_results_to_server(timestamp, prediction, direction)


# Example usage
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
prediction = "员工A"  # Example prediction, replace with actual prediction logic
direction = "in"  # Example direction, replace with actual direction logic

save_results_to_file_and_server([timestamp, prediction, direction])