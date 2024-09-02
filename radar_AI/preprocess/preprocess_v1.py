from scipy import signal
from sktime.libs.vmdpy import VMD
import omegaconf

from radar_AI.preprocess.module.fft_spectrum import fft_spectrum
from radar_AI.preprocess.module.fft_spectrum import range_doppler_angle_fft
from radar_AI.preprocess.module.Peakcure import peakcure
from datetime import datetime
from collections import deque
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import os
import sys
import mne


"""
To Preprocess Raw Data from RADAR to apply to Deep Learning
"""
class preprocess_v1:
    def __init__(self, 
                 radar : omegaconf.DictConfig, 
                 preprocessor : omegaconf.DictConfig):
        self.num_samples_per_chirp = radar.num_samples_per_chirp
        self.num_chirps_per_frame = 80 # radar.num_chirps_per_frame
        self.rx_mask = radar.rx_mask

        self.preprocessor = preprocessor

        self.range_window = signal.windows.blackmanharris(self.num_samples_per_chirp).reshape(1, self.num_samples_per_chirp)
        self.doppler_window = signal.windows.blackmanharris(self.num_chirps_per_frame).reshape(1, self.num_chirps_per_frame)
        self.angle_window = signal.windows.blackmanharris(self.rx_mask).reshape(1, self.rx_mask)

    def preprocess(self, data_in):
        range_fft = self.range_fft(data_in)
        phase_unwrap = self.unwrap(range_fft)
        u = self.vmd(phase_unwrap)[0]

        return u
    
    def range_fft(self, data):
        range_fft = fft_spectrum(data, self.range_window)
        return range_fft

    # input with "range_fft", output : same format
    def doppler_fft(self, data):
        doppler_fft = fft_spectrum(data.T, self.doppler_window)
        return doppler_fft.T
    
    def unwrap(self, data):
        phase_unwrap = peakcure(data)[2]
        return phase_unwrap
    
    def vmd(self, data):
        u, u_hat, omega = VMD(data, self.preprocessor.alpha, self.preprocessor.tau, self.preprocessor.K, self.preprocessor.DC, self.preprocessor.init, self.preprocessor.tol)
        return u, u_hat, omega

    def angle_range_doppler_fft(self, data):
        range_fft = range_doppler_angle_fft(data, self.range_window)
        doppler_fft = range_doppler_angle_fft(range_fft.transpose(2,0,1), self.doppler_window)
        angle_fft = range_doppler_angle_fft(doppler_fft.transpose(2,0,1), self.angle_window)
        return angle_fft.transpose(2,0,1)


"""
Function to filtering & interpolating Respiratory value from PSG
"""
def interpolate_resp(resp):
    mask = resp >= 20
    resp[mask] = np.nan
    mask = resp <= 0
    resp[mask] = np.nan
    
    nans, x = np.isnan(resp), lambda z: z.nonzero()[0]
    resp[nans] = np.interp(x(nans), x(~nans), resp[~nans])
    return resp
    
def process_segment(args):
    concatenated_data, second, second_diff, preprocess_module = args
    return preprocess_module.preprocess(concatenated_data[(second + second_diff + 6)*20 : (second + second_diff + 10)*20])

def process_data(data_list, second_diff, preprocess_module):
    processed_data = []

    tasks = []
    for i in range(len(data_list) - 2):
        concatenated_data = np.concatenate(data_list[i:i+2], axis=0)
        for second in range(0, 60, 10):
            tasks.append((concatenated_data, second, second_diff, preprocess_module))

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_segment, tasks)

    processed_data.extend(results)

    return processed_data[:-1]


"""
Make Input & Output data with preprocessed data.
Split data from 23:03 ~ 05:27 data.
"""
def make_input_output(folder_name, preprocess_module):
    directory = 'data/' + folder_name + '/radar'
    
    start_file = folder_name.split('_')[0] + '_2304.npy'
    end_file = str(int(folder_name.split('_')[0]) + 1) + '_0529.npy' # +1분을 해야함
    all_files = sorted([f for f in os.listdir(directory) if f.endswith('.npy')])
    filtered_files = [f for f in all_files if start_file <= f <= end_file]
    data_list = []
    for file in filtered_files:
        file_path = os.path.join(directory, file)
        data = np.load(file_path)
        data_list.append(data)

    file_path = './data/' + folder_name + '/' + folder_name + '.xls'
    df = pd.read_excel(file_path, engine='xlrd', sheet_name = 1)
    new_column_names = {
        'TimeStamp': 'time',
        'HeartRate_PeakToPeak_10s_': 'HR_p2p',
        'HeartRate_Mean_10s_': 'HR_mean',
        'Sleep' : 'Stage',
        'Events_Movement__Count_' : 'movemnet'
    }
    df.rename(columns=new_column_names, inplace=True)
    df = df.iloc[1:]
    
    df['time'] = pd.to_datetime(df['time'])
    
    df = df[(df['time'].dt.time >= datetime.strptime('23:03', '%H:%M').time()) |
            (df['time'].dt.time <= datetime.strptime('05:27', '%H:%M').time())]
    
    # Stage를 숫자로 매핑하는 딕셔너리 생성
    stage_mapping = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    
    # Stage 열의 값을 숫자로 변환
    df['Stage'] = df['Stage'].map(stage_mapping)
    
    second_diff = df['time'].iloc[0].second
    
    df = df[1:]
    
    heart_label = torch.tensor(np.array(df['HR_mean'].astype('float')), dtype = torch.float32)
    sleep_stage_label = torch.tensor(df['Stage'].values)

    
    # Example usage
    processed_data = process_data(data_list, second_diff, preprocess_module)
    

    if (df['time'].iloc[0].time() >= datetime.strptime('23:03:20', '%H:%M:%S').time() and
        df['time'].iloc[-1].time() <= datetime.strptime('23:02:50', '%H:%M:%S').time()):
        raise Exception("Error! [Start, End Time is not correct!]")
    elif (df['time'].iloc[0].time() >= datetime.strptime('23:03:20', '%H:%M:%S').time()):
        processed_data = processed_data[len(heart_label) - len(df):]
    elif (df['time'].iloc[-1].time() <= datetime.strptime('23:02:50', '%H:%M:%S').time()):
        processed_data = processed_data[:len(df)]


    """ MAKE RESP LIST """
    file = './data/' + folder_name + '/' + folder_name + '.edf'
    data = mne.io.read_raw_edf(file)
    target_time_naive = datetime.strptime('23:03', '%H:%M')
    meas_start_time = data.info['meas_date']
    target_time = target_time_naive.replace(year=meas_start_time.year, 
                                            month=meas_start_time.month, 
                                            day=meas_start_time.day, 
                                            tzinfo=meas_start_time.tzinfo)
    time_diff = (target_time - meas_start_time).total_seconds()
    adjusted_time_diff = time_diff + second_diff
    resp_data = data['Resp Rate'][0][0][int(adjusted_time_diff) * 25:]
    resp = interpolate_resp(resp_data)
    resp_mean = []
    for i in range(len(processed_data)):
        resp_mean.append(np.mean(resp[25*10*i:25*10*(i+1)]))
    resp_mean = torch.tensor(resp_mean, dtype = torch.float32)
        

    return processed_data, heart_label, sleep_stage_label, resp_mean



"""
Function to preprocess the output from HR, RR Model
To apply data to Sleep Model directly
"""
def create_sequence(heart_model, resp_model, trainloader, device):
    q = deque(maxlen = 30)
    input_list = []
    target_list = []
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)

        HR = heart_model(inputs).detach()
        RR = resp_model(inputs).detach()
        HR_RR = torch.cat((HR, RR), dim=1).unsqueeze(0).to('cpu')
        q.append(HR_RR.reshape(256))

        if (len(q) == 30):
            input = torch.tensor(np.array(q))
            input_list.append(input.cpu().numpy())
            target_list.append(targets.cpu().numpy())

    return np.array(input_list), np.array(target_list)


"""
Provide DataLoader about HR, RR, Sleep only using this class
"""
class data_provider:
    def __init__(self, preprocess_module, device):    
        # To Make DataLoader
        current_directory = os.path.dirname(os.path.abspath(__file__))
        utils_directory = os.path.join(current_directory, '..', 'utils')
        sys.path.append(utils_directory)
        import radar_AI.utils.dataloader as dataloader
        self.dataloader = dataloader
        ###
        
        directory_path = 'data/'
    
        exclude_list = ['20240825_LGH_57344_wait', 'EX_HR_3RX', '20240806_CGW', '.ipynb_checkpoints', '20240813_LGH_57345_side', '20240813_LGH_57344',
                       '20240827_CJM_57344', '20240827_MYJ_57345', '20240828_MYJ_57345', '20240828_PDJ_57344', '20240829_CJM_57345', '20240829_PDJ_57344','*.zip']
    
        folder_names = [name for name in os.listdir(directory_path) 
                    if os.path.isdir(os.path.join(directory_path, name)) and name not in exclude_list]

        self.device = device
        processed_data=[]
        heart_label=[]
        resp_label=[]
        sleep_stage_label=[]
        for folder in folder_names:
            data = make_input_output(folder, preprocess_module)
            processed_data.append(data[0])
            heart_label.append(data[1])
            sleep_stage_label.append(data[2])
            resp_label.append(data[3])
    
        self.processed_data = np.concatenate(processed_data, axis = 0)
        self.heart_label = np.concatenate(heart_label, axis = 0)
        self.resp_label = np.concatenate(resp_label, axis = 0)
        self.sleep_stage_label = np.concatenate(sleep_stage_label, axis = 0)
    

    
    def make_hr_dataloader(self, split_ratio = 0.2):
        train_test_split = int(len(self.processed_data)*split_ratio)
    
        trainloader_hr = self.dataloader.train_dataloader(torch.tensor(self.processed_data[:train_test_split]).unsqueeze(1).float(), self.heart_label[:train_test_split], 2**8)
        testloader_hr = self.dataloader.test_dataloader(torch.tensor(self.processed_data[train_test_split:]).unsqueeze(1).float(), self.heart_label[train_test_split:])
        return trainloader_hr, testloader_hr

    def make_rr_dataloader(self, split_ratio = 0.2):
        train_test_split = int(len(self.processed_data)*split_ratio)
    
        trainloader_rr = self.dataloader.train_dataloader(torch.tensor(self.processed_data[:train_test_split]).unsqueeze(1).float(), torch.tensor(self.resp_label[:train_test_split]).float(), 2**8)
        testloader_rr = self.dataloader.test_dataloader(torch.tensor(self.processed_data[train_test_split:]).unsqueeze(1).float(), torch.tensor(self.resp_label[train_test_split:]).float())
        return trainloader_rr, testloader_rr

    def make_sleep_dataloader(self, heart_rate_model_extractor, respiration_model_extractor, split_ratio = 0.2):
        train_test_split = int(len(self.processed_data)*split_ratio)

        trainloader_sleep = self.dataloader.train_dataloader(torch.tensor(self.processed_data[:train_test_split]).unsqueeze(1).float(), torch.LongTensor(self.sleep_stage_label[:train_test_split]), 1, False)
        testloader_sleep = self.dataloader.test_dataloader(torch.tensor(self.processed_data[train_test_split:]).unsqueeze(1).float(), torch.LongTensor(self.sleep_stage_label[train_test_split:])) 

        sleep_train_input, sleep_train_label = create_sequence(heart_rate_model_extractor, respiration_model_extractor, trainloader_sleep, self.device)
        sleep_test_input, sleep_test_label = create_sequence(heart_rate_model_extractor, respiration_model_extractor, testloader_sleep, self.device)
    
        trainloader_sleep_preprocessed = self.dataloader.train_dataloader(torch.tensor(sleep_train_input), torch.LongTensor(sleep_train_label), 1)
        testloader_sleep_preprocessed = self.dataloader.test_dataloader(torch.tensor(sleep_test_input), torch.LongTensor(sleep_test_label))

        del trainloader_sleep, testloader_sleep

        return trainloader_sleep_preprocessed, testloader_sleep_preprocessed

