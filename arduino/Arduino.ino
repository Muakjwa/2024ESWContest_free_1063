int motorPin = 3;
int speakerPin = 7;

unsigned long previousSpeakerMillis = 0;
unsigned long previousMotorMillis = 0;

int speakerRepeatCount = 0;
int motorRepeatCount = 0;

unsigned long speakerOnTime = 0;
unsigned long speakerOffTime = 0;
int speakerMaxRepeats = 0;

unsigned long motorOnTime = 0;
unsigned long motorOffTime = 0;
int motorMaxRepeats = 0;

void setup() {
  pinMode(motorPin, OUTPUT);
  pinMode(speakerPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();

    // 새로운 명령을 받을 때마다 카운터와 상태 초기화
    speakerRepeatCount = 0;
    motorRepeatCount = 0;

    // # A : 스티어링 휠을 10초간 파지하지 않았을 경우
    if (receivedChar == 'A') {
        setSpeakerControl(300, 300, 5);
    } 
    // # B : 스티어링 휠을 20초간 파지하지 않았을 경우
    else if (receivedChar == 'B') {
        setSpeakerControl(400, 200, 1000);
    }
    // # C : 졸음감지가 2~5초간 지속이 되었을 경우 
    else if (receivedChar == 'C') {
        setMotorControl(250, 250, 6);
    } 
    // # D : 졸음감지가 5초 이상이 지속이 되었을 경우
    else if (receivedChar == 'D') {
        setMotorControl(100, 100, 1000);
    } 
    // # E : 핸들을 파지하지 않은 상태에서 졸음감지가 2~5초간 지속이 되었을 경우 
    else if (receivedChar == 'E') {
        setSpeakerControl(300, 300, 5);
        setMotorControl(250, 250, 6);
    } 
    // # F : 핸들을 파지하지 않은 상태에서 졸음감지가 5초 이상 지속이 되었을 경우 
    else if (receivedChar == 'F') {
        setSpeakerControl(400, 200, 1000);
        setMotorControl(100, 100, 1000);
    } 
    // # 정상 주행 상태 (스피커: OFF / 진동모터: OFF)
    else if (receivedChar == 'Z') {
      digitalWrite(speakerPin, LOW);
      digitalWrite(motorPin, LOW);
      speakerMaxRepeats = 0;
      motorMaxRepeats = 0;
    }
  }

  // 스피커와 모터 제어를 비동기적으로 실행
  controlSpeaker();
  controlMotor();
}

// 스피커 제어 파라미터 설정 함수
void setSpeakerControl(unsigned long onTime, unsigned long offTime, int repeat) {
  speakerOnTime = onTime;
  speakerOffTime = offTime;
  speakerMaxRepeats = repeat;
  speakerRepeatCount = 0; // 새 명령이 들어오면 반복 카운트를 초기화
  previousSpeakerMillis = millis(); // 타이머 초기화
}

// 모터 제어 파라미터 설정 함수
void setMotorControl(unsigned long onTime, unsigned long offTime, int repeat) {
  motorOnTime = onTime;
  motorOffTime = offTime;
  motorMaxRepeats = repeat;
  motorRepeatCount = 0; // 새 명령이 들어오면 반복 카운트를 초기화
  previousMotorMillis = millis(); // 타이머 초기화
}

// 스피커 제어 함수
void controlSpeaker() {
  unsigned long currentMillis = millis();
  
  if (speakerRepeatCount < speakerMaxRepeats) {
    if (digitalRead(speakerPin) == LOW && currentMillis - previousSpeakerMillis >= speakerOffTime) {
      digitalWrite(speakerPin, HIGH);
      previousSpeakerMillis = currentMillis;
    } else if (digitalRead(speakerPin) == HIGH && currentMillis - previousSpeakerMillis >= speakerOnTime) {
      digitalWrite(speakerPin, LOW);
      previousSpeakerMillis = currentMillis;
      speakerRepeatCount++;
    }
  }
}

// 모터 제어 함수
void controlMotor() {
  unsigned long currentMillis = millis();
  
  if (motorRepeatCount < motorMaxRepeats) {
    if (digitalRead(motorPin) == LOW && currentMillis - previousMotorMillis >= motorOffTime) {
      digitalWrite(motorPin, HIGH);
      previousMotorMillis = currentMillis;
    } else if (digitalRead(motorPin) == HIGH && currentMillis - previousMotorMillis >= motorOnTime) {
      digitalWrite(motorPin, LOW);
      previousMotorMillis = currentMillis;
      motorRepeatCount++;
    }
  }
}
