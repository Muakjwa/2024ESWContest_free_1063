# DMS (Driver Monitoring System) based on FMCW Radar & Capacitive Touch Sensor
![real_test](https://github.com/user-attachments/assets/35773330-7876-49be-815c-a971e254cf13)

<br>

## 프로젝트 소개
- 운전자 상태 분석 시스템은 레이더 센서를 이용해 운전자의 심박수, 호흡, 수면 상태를 비접촉식으로 감지합니다. 
- 스티어링 휠을 통해 운전자가 운전대를 잡고 있는지, 한 손인지 두 손인지, 장갑을 착용했는지를 인식합니다. 
- 감지된 정보를 바탕으로 진동과 경고음을 통해 운전에 집중하도록 피드백을 제공합니다. 
- 이를 통해 운전자의 안전한 운전을 돕는 스마트 시스템입니다.
<br>
<img width="748" alt="alarm_principle" src="https://github.com/user-attachments/assets/305e1c9a-682d-43ae-81a2-c6803cca21ef">


## Team Member

<div align="center">

| **김대원** | **유선우** | **이재현** | **최기원** |
| :------: |  :------: | :------: | :-----: |
| DGIST | DGIST | DGIST | DGIST |
| Mechatronics | Computer Engineering | Computer Engineering | NeuroSicence |
| [<img src="https://avatars.githubusercontent.com/u/176874399?v=4" height=150 width=150> <br/> @dw622](https://github.com/dw622) | [<img src="https://avatars.githubusercontent.com/u/94523712?s=400&v=4" height=150 width=150> <br/> @Muakjwa](https://github.com/Muakjwa) | [<img src="https://avatars.githubusercontent.com/u/149148774?v=4" height=150 width=150> <br/> @monkcat](https://github.com/monkcat) | [<img src="https://avatars.githubusercontent.com/u/149148774?v=4" height=150 width=150> <br/> @monkcat] |
</div>
<br>

## 1. 개발환경

### Hardware
- Radar : cysbsyskit-dev-01(mcu) + bgt60tr13c(radar) [infineon]
- Steering Wheel : PSoC4100S Max pioneer kit(mcu) + BMW 520d(wheel) [infineon]
- Main Computer : Raspberry Pi 5
- Feedback Controller : Arduino Uno R4
- Feedback Device : Buzzer, DC Motor

### Software
- Radar Build : ModusToolBox
- Steering Wheel Build : ModusToolBox
- Overall Architecture : Python
- AI Model : Pytorch, Scikit-Learn
- Version & Issue Management : Github

## 2. 프로젝트 구조

```

├── .gitignore
├── GUI
│   ├── gui.py
│   ├── gui.ui
├── main.py
├── model
│   ├── hr_model.pt
│   ├── rr_model.pt
│   ├── wheel_model_v0.pkl
├── radar.py
├── radar_AI
│   ├── .gitignore
│   ├── config
│   │   ├── preprocess
│   │   │   ├── v1.yaml
│   │   ├── radar
│   │   │   ├── HR.yaml
│   ├── model
│   │   ├── model_v0.py
│   │   ├── model_v1.py
│   │   ├── trainer.py
│   ├── preprocess
│   │   ├── module
│   │   │   ├── Diffphase.py
│   │   │   ├── IIR_Breath.py
│   │   │   ├── IIR_Heart.py
│   │   │   ├── PeakBreath.py
│   │   │   ├── PeakHeart.py
│   │   │   ├── Peakcure.py
│   │   ├── preprocess_v1.py
│   ├── train_v1.py
│   ├── utils
│   │   ├── dataloader.py
│   │   ├── plot_utils.py
├── steering_wheel
│   ├── steering_wheel
│   │   ├── README.md
│   │   ├── collect_data.py
│   │   ├── model
│   │   │   ├── HOD_ML_VER0(LogisticRegression).pkl
│   │   │   ├── WHEEL_VER0.pkl
│   │   ├── steering_wheel.py
├── wheel_module.py
├── file_descriptor.py
```

## 3. 역할 분담

### 김대원
- Build MCU Program
- Steering Wheel AI Model (Hands on, Hands off, One Hand, Two Hands, etc.)
- Visualization of Heart, Respiratory Wave
- Hardware Setting

### 유선우
- Radar AI Model (HeartRate, RespiratoryRate, Sleep)
- Radar data Visualization
- Overall System Code Integration
- Collecting PSG + Radar Dataset

### 이재현
- Designing and Programming GUI
- Setting Case of each situation
- Intergrating Hardware + Software
- Test & Debugging

### 최기원
- Visualization of Radar
- Collecting PSG + Radar Dataset
- Test & Debugging