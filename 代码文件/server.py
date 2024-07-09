from flask import Flask, request, jsonify, render_template_string
import csv
from datetime import datetime

app = Flask(__name__)


@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    timestamp = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
    prediction = data['prediction']
    direction = data['direction']

    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, prediction, direction])

    return jsonify({'status': 'success'}), 200


@app.route('/show_results')
def show_results():
    results = []
    with open('results.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            results.append(row)

    # 加载 HTML 文件内容
    with open('results.html', 'r') as htmlfile:
        html_content = htmlfile.read()

    # 使用 render_template_string 渲染 HTML 内容
    return render_template_string(html_content, results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)