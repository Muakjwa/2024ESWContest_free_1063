import serial
import pandas as pd


class data_collector():
    def __init__(self, filename, port = 'COM10', baudrate = 115200, timeout = 1):
        # 시리얼 연결 설정
        self.ser = serial.Serial() 
        self.ser.port = port
        self.ser.baudrate = baudrate
        self.ser.timeout = timeout

        self.ser.close() # 에러 방지용
        self.ser.open()

        # 데이터를 저장할 리스트 생성
        self.data_list = []

    def collect(self):
        try:
            while True:
                data = self.ser.readline().decode('utf-8').strip()  # serial에 출력되는 데이터를 읽어들임
                if data:
                    results = list(map(int, data.split(",")))
                    print(results)
                    self.data_list.append(results)

        except KeyboardInterrupt:
            print("Stopped by user.")

        finally:
            self.ser.close()

            # 수집한 데이터를 DataFrame으로 변환
            new_df = pd.DataFrame(self.data_list, columns=["index", "Sensor", "Shield"])

            # Baseline 컬럼 초기화
            new_df['Sensor_Baseline'] = new_df['Sensor']
            new_df['Shield_Baseline'] = new_df['Shield']
            new_df['Sensor_changes'] = 0
            new_df['Shield_changes'] = 0
            new_df['Diff'] = 0
            new_df['Ratio'] = 0

            # 상태 추적 변수 초기화
            off_streak = 49  # 첫 50개의 데이터에 대해 기본값을 적용
            under_threshold_streak = 0
            baseline_sensor = None
            baseline_shield = None
            allow_baseline_update = True

            # Baseline 계산을 위한 초기 설정
            baseline_sensor = new_df['Sensor'][:50].mean()
            baseline_shield = new_df['Shield'][:50].mean()

            for i in range(50, len(new_df)):
                sensor_noise = abs(new_df.loc[i, 'Sensor'] - new_df.loc[i-1, 'Sensor'])
                shield_noise = abs(new_df.loc[i, 'Shield'] - new_df.loc[i-1, 'Shield'])

                if sensor_noise <= 3000 and shield_noise <= 20000:
                    off_streak += 1
                    under_threshold_streak += 1

                    if off_streak >= 50 and allow_baseline_update:
                        # Baseline 업데이트
                        baseline_sensor = new_df['Sensor'][i-49:i+1].mean()  # 50개의 Sensor 평균
                        baseline_shield = new_df['Shield'][i-49:i+1].mean()  # 50개의 Shield 평균
                        off_streak = 0  # 초기화
                else:
                    off_streak = 0
                    under_threshold_streak = 0

                if baseline_sensor is not None and baseline_shield is not None:
                    sensor_diff = new_df.loc[i, 'Sensor'] - baseline_sensor
                    shield_diff = new_df.loc[i, 'Shield'] - baseline_shield

                    new_df.loc[i, 'Sensor_changes'] = sensor_diff
                    new_df.loc[i, 'Shield_changes'] = shield_diff
                    new_df.loc[i, 'Diff'] = shield_diff - sensor_diff

                    if sensor_diff > 3000 or shield_diff > 20000:
                        # Baseline 업데이트 비활성화
                        allow_baseline_update = False
                        under_threshold_streak = 0
                        new_df.loc[i, 'Ratio'] = shield_diff / sensor_diff if sensor_diff != 0 else 0
                    else:
                        new_df.loc[i, 'Ratio'] = 0
                        if under_threshold_streak >= 10:
                            allow_baseline_update = True

                    # Baseline 적용
                    new_df.loc[i, 'Sensor_Baseline'] = baseline_sensor
                    new_df.loc[i, 'Shield_Baseline'] = baseline_shield

            # 처리된 데이터를 엑셀 파일로 저장
            new_df.to_excel('HOD_DATA_선우1.xlsx', index=False)

            print("HOD_DATA_선우1.xlsx에 데이터 저장 완료")


if __name__ == "__main__":
    # data 수집 객체 생성
    # (connecting steering wheel & set the parameter)
    data_provider = data_collector()
    # data 수집 시작
    data_provider.collect()