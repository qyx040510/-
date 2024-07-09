import time
import os
import cv2  

# GPIO路径
GPIO_PATH = "/sys/class/gpio"

# GPIO引脚定义
LED1_PIN = "45"
LED2_PIN = "40"
BUZZER_PIN = "44"

# 导出GPIO
def export_gpio(pin):
    export_path = os.path.join(GPIO_PATH, "export")
    if not os.path.exists(os.path.join(GPIO_PATH, f"gpio{pin}")):
        with open(export_path, 'w') as f:
            f.write(pin)

# 设置GPIO方向
def set_gpio_direction(pin, direction):
    direction_path = os.path.join(GPIO_PATH, f"gpio{pin}/direction")
    with open(direction_path, 'w') as f:
        f.write(direction)

# 设置GPIO值
def set_gpio_value(pin, value):
    value_path = os.path.join(GPIO_PATH, f"gpio{pin}/value")
    with open(value_path, 'w') as f:
        f.write(str(value))

# 初始化GPIO
def init_gpio(pin, direction):
    export_gpio(pin)
    time.sleep(0.1)  # 给系统一些时间来创建GPIO文件夹
    set_gpio_direction(pin, direction)

# 拍摄照片
def take_photo():
    camera = cv2.VideoCapture(0) 
    ret, frame = camera.read()
    if ret:
        cv2.imwrite("photo.jpg", frame)
    camera.release()

# 主函数
def main(result):
    # 初始化GPIO引脚
    init_gpio(LED1_PIN, "out")
    init_gpio(LED2_PIN, "out")
    init_gpio(BUZZER_PIN, "out")
    
    if result == 0:
        # LED1亮五秒
        set_gpio_value(LED1_PIN, 1)
        # 蜂鸣器响一秒
        set_gpio_value(BUZZER_PIN, 1)
        time.sleep(1)
        set_gpio_value(BUZZER_PIN, 0)
        # 拍摄照片
        take_photo()
        time.sleep(4)  # 剩余时间保持LED1亮
        set_gpio_value(LED1_PIN, 0)
    else:
        # LED2亮五秒
        set_gpio_value(LED2_PIN, 1)
        time.sleep(5)
        set_gpio_value(LED2_PIN, 0)

