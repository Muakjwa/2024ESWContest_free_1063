import numpy as np
from collections import deque
import serial
import threading
import time
import joblib
import pandas as pd

# UART 설정
ser_wheel = serial.Serial('com10', 115200, timeout=1)   # 포트 이름과 보드레이트를 필요에 따라 변경
uart_buffer = deque(maxlen = 1)

lock = threading.Lock()

# UART 데이터를 읽어오는 스레드
def read_uart_wheel():
    i = 0
    while True:
        if ser_wheel.in_waiting > 0:
            line = ser_wheel.readline().decode('utf-8').strip()
            if line:
                try:
                    index, sensor, shield = map(int, line.split(','))
                    uart_buffer.append([index, sensor, shield])
                except ValueError:
                    pass  # 잘못된 데이터는 무시
    time.sleep(0.01)

model = joblib.load('./model/wheel_model_v0.pkl')

def sensor_diff(sensor):
    if sensor >= 30000:
        sensor_change = sensor - 22000
    else:
        sensor_change = 0
    return sensor_change

def shield_diff(shield):
    if shield >= 160000:
        shield_change = shield - 135000
    else:
        shield_change = 0
    return shield_change

def model_prediction(sensor,shield):
    input_data = pd.DataFrame([[sensor, shield]], columns=['Sensor_changes', 'Shield_changes'])
    prediction = model.predict(input_data)
    return int(prediction[0])

def conductivity_cal(sensor,shield):
    if sensor == 0 or shield == 0:
        conductivity = 0
    else:
        conductivity = round(((shield-sensor)/120000)*100)
    return 100 - conductivity

def process_data_wheel(queue):
    while True:
        with lock:
            if len(uart_buffer) > 0:
                index, sensor, shield = uart_buffer[0]
                xdata = index // 37000  # 500에서 1000까지의 xdata 생성
                sensor = sensor_diff(sensor)
                shield = shield_diff(shield)
                conductivity = conductivity_cal(sensor,shield)
                prediction = model_prediction(sensor, shield)
                queue.put((xdata, sensor, shield, conductivity, prediction))
            else:
                queue.put((0,0,0,0,0))

        time.sleep(0.1)  # 0.05초 대기
