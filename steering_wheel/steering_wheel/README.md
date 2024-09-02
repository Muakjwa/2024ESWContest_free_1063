## HOD(Hands-Off Detection) System
##### This study focuses on a simple machine learning model applied to the __HOD (Hands-On Detection) system__, aiming to determine the material and contact area on the steering wheel using machine learning. The steering wheel handle is composed of two conductive fabric layers, each connected to __Sensor__ and __Shield__. The capacitance values measured from the electrodes are set as the basic parameters, and the model seeks to make judgments based on the difference or ratio of changes between these two values.
#### The parameters are as follows:
1. Capacitance of the Sensor electrode
2. Capacitance of the Shield electrode
3. Change in the Sensor electrode
4. Change in the Shield electrode
5. Difference between the changes
6. Ratio between the changes

#### Using these parameters, the model aims to determine the following factors:
1. Grip status: Hands-On/Hands-Off
2. Touch type: Single hand, both hands, or just touch
3. Contact material: Bare hand or leather glove
4. Steering status: Straight driving or steering


## Data Collection
현재 데이터 수집 방식에 대한 설명
1. UART를 이용하여 실시간으로 Sensor와 Shield에 대한 정보를 받아옴.
2. Sensor와 Shield의 Baseline을 잡기 위한 방법
    - 우선, 노이즈의 범위를 지정하여 노이즈로 생각되는 범위내에서만 움직이면 파지하지 않은 것으로 간주
    - 변화량을 계산하였을 때, 일정값을 넘어간다면 파지 중으로 판단하고 Baseline 업데이트 중단
3. Baseline을 토대로 각 Sensor와 Shield에 대한 변화량을 계산함
4. 변화향을 토대로 Diff와 Ratio를 계산
    - Diff : Shield의 변화량 -  Sensor의 변화량
    - Ratio : Shield의 변화량 / Sensor의 변화량
    
    
### 라벨 설정
 1. 파지 여부: hands-on (0) /  hands-off (1)
2. 접촉 면적 : 없음 (0) /  한손 (1) /  양손(2)        *추후 터치에 관한 내용 추가 예정(한손도 양손도 아닌 경우)
3. 접촉 물질 : 없음 (0) /  맨손 (1) /  가죽 장갑 (2)  *추후 다른 장갑에 관한 내용 추가 예정(면장갑, 목장갑)

|최종 라벨|파지 여부|접촉 면적|접촉 물질|
|------|------|------|------|
|0|0|0|0|
|1|1|2|1|
|2|1|1|1|
|3|1|2|2|
|4|1|1|2|