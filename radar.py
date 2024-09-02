import numpy as np
from scipy.signal import butter, sosfilt
from collections import deque
import serial
import threading
import time

import torch
from radar_AI.preprocess import preprocess_v1
from radar_AI.model.model_v1 import BasicBlock
from radar_AI.model.model_v1 import ResNet
from radar_AI.model.model_v1 import SleepStageLSTM, SleepStageGRU
from omegaconf import OmegaConf

# UART 설정
ser_radar = serial.Serial('com8', 115200, timeout=1) # 포트 이름과 보드레이트를 필요에 따라 변경
uart_buffer = deque(maxlen=400)  # UART로부터 읽은 데이터를 저장하는 버퍼

lock = threading.Lock()

# UART 데이터를 읽어오는 스레드
def read_uart():
    while True:
        if ser_radar.in_waiting > 0:
            line = ser_radar.readline().decode('utf-8').strip()
            if line:
                try:
                    data_point = float(line)
                    with lock:
                        uart_buffer.append(data_point)
                except ValueError:
                    pass  # 잘못된 데이터는 무시
        # time.sleep(0.01)

def irr_breath(phase_remove):
    fs = 20 
    f1_breath = (8/60) / (fs/2)
    f2_breath = (20/60) / (fs/2) 
    sos_breath = butter(4, [f1_breath, f2_breath], btype='bandpass', output='sos')
    res_breath = sosfilt(sos_breath, phase_remove)
    return res_breath

def irr_heart(phase_remove):
    fs = 20 
    f1_heart = (40/60) / (fs/2) 
    f2_heart = (100/60) / (fs/2)
    sos_heart = butter(8, [f1_heart, f2_heart], btype='bandpass', output='sos')
    res_heart = sosfilt(sos_heart, phase_remove)
    return res_heart

def process_data(queue):
    diff_queue = deque(maxlen=20)
    while True:
        with lock:
            if len(uart_buffer) >= 400:
                phase = np.array(list(uart_buffer))
                xdata = np.array(list(range(200))) 
                breathdata = irr_breath(phase)[399]
                heartdata = irr_heart(phase)[399]
                diff = np.max(phase[-50:]) - np.min(phase[-50:])
                diff_queue.append(diff)
                presence = np.sum(np.array(list(diff_queue)) >= 0.5) >=5
                
                queue.put((int(xdata[0]), float(breathdata), float(heartdata), presence))
            else:
                queue.put((0, 0, 0, 0))
    
        time.sleep(0.02) 
        
def process_AI(queue):
    hr_model = ResNet(BasicBlock, [2, 2], 0, 1)
    rr_model = ResNet(BasicBlock, [2, 2], 0, 1)
    hr_fc = ResNet(BasicBlock, [2, 2], 0)
    rr_fc = ResNet(BasicBlock, [2, 2], 0)
    sleep_model = SleepStageGRU(input_size = 256, hidden_size=64, num_layers=2, num_classes=5)
    HR_config = OmegaConf.load('radar_AI/config/radar/HR.yaml')
    preprocess_v1_config = OmegaConf.load('radar_AI/config/preprocess/v1.yaml')
    preprocess_module = preprocess_v1.preprocess_v1(HR_config.config, preprocess_v1_config.config)

    hr_model.load_state_dict(torch.load('./model/hr_model.pt', map_location=torch.device('cpu'), weights_only=True))
    rr_model.load_state_dict(torch.load('./model/rr_model.pt', map_location=torch.device('cpu'), weights_only=True))
    hr_fc.load_state_dict(torch.load('./model/hr_model.pt', map_location=torch.device('cpu'), weights_only=True))
    rr_fc.load_state_dict(torch.load('./model/rr_model.pt', map_location=torch.device('cpu'), weights_only=True))
    sleep_model.load_state_dict(torch.load('./model/sleep_model.pt', map_location=torch.device('cpu'), weights_only=True))

    sleep = deque(maxlen = 30)
    sleep_warning = deque(maxlen = 5)
    drowsiness = 0 
    while True:
        if len(uart_buffer) >= 80:
            u = torch.tensor(preprocess_module.vmd(np.array(np.array(uart_buffer)[-80:]))[0].reshape(1, 1, 7, 80), dtype = torch.float32)
            hr_latent = hr_model(u)
            rr_latent = rr_model(u)
            hr = hr_fc.fc(hr_latent)
            rr = rr_fc.fc(rr_latent)
            sleep_concat = torch.cat((hr_latent, rr_latent), dim = 1).reshape(256)
            sleep.append(sleep_concat.detach().numpy())
            sleep_warning.append(0)
            if (len(sleep) == 30):
                sleep_stage = sleep_model(torch.tensor(np.array(sleep)).unsqueeze(0))
                print(np.argmax(sleep_stage.detach().numpy()))
                sleep_warning.append(int(~~np.argmax(sleep_stage.detach().numpy())))
                if np.sum(np.array(sleep_warning)) >= 4:
                    drowsiness = 2
                elif np.sum(np.array(sleep_warning)[-2:]) == 2:
                    drowsiness = 1
                
            
            if (len(sleep) == 30):
                queue.put((int(hr[0][0].detach()), int(rr[0][0].detach()), np.argmax(sleep_stage.detach().numpy()), drowsiness))
            else:
                queue.put((int(hr[0][0].detach()), int(rr[0][0].detach()), 1, 0))
                
            time.sleep(1)