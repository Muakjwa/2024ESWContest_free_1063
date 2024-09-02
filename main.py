import sys
import threading
from queue import Queue
import numpy as np
from collections import deque
import serial 

from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QProgressBar
from PySide6.QtGui import QPixmap, QMovie
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer

from GUI.gui import Ui_Sensor
from radar import process_data, read_uart, process_AI
from wheel_module import read_uart_wheel, process_data_wheel


class MainWindow(QMainWindow, Ui_Sensor):
    def __init__(self, queue, queue_wheel, queue_AI):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # UI 초기화
        
        # Arduino Info
        self.feedback  = serial.Serial('com9', 9600, timeout=1)
        self.grip_stat = 0
        self.drowsiness_stat = -1

        ###
        self.presence = 0
        self.grip = 0
        self.grip_queue = deque(maxlen = 200)
        self.drowsiness = 0
        self.drowsiness_state = 0
        self.sleep_stage_short = deque(maxlen = 2) #2s decision
        self.sleep_stage_long = deque(maxlen = 5)  #5s decision

        # Queue For Data Transmitting
        self.queue = queue
        self.queue_wheel = queue_wheel
        self.queue_AI = queue_AI

        # real-time graph
        self.xdata_radar = list(range(200))  # 처음 xdata는 0~49
        self.ydata_radar = [0]*200  # 초기 ydata
        self.ydata_radar_2 = [0]*200  # 초기 ydata
        
        self.xdata = list(range(200))  # 처음 xdata는 0~49
        self.ydata = [0]*200  # 초기 ydata
        self.ydata2 = [0]*200  # 초기 ydata2

        # radar graph
        self.plotWidget_1 = pg.PlotWidget()
        self.plotWidget_2 = pg.PlotWidget()
        self.plotWidget_1.setBackground('w')
        self.plotWidget_2.setBackground('w')

        self.plot1 = self.plotWidget_1.plot(self.xdata, self.ydata, pen=pg.mkPen('red', width=3))  # y1 데이터를 위한 플롯
        self.plot2 = self.plotWidget_2.plot(self.xdata, self.ydata, pen=pg.mkPen('blue', width=3))  # y2 데이터를 위한 플롯
        
        # x축과 y축 숨기기
        self.plotWidget_1.getPlotItem().hideAxis('left')   # y축 숨기기
        self.plotWidget_1.getPlotItem().hideAxis('bottom') # x축 숨기기
        self.plotWidget_1.setYRange(-1.0, 1.0)
        self.plotWidget_1.setXRange(0, 200)
        self.plotWidget_1.enableAutoRange(axis='xy', enable=False)
        # x축과 y축 숨기기
        self.plotWidget_2.getPlotItem().hideAxis('left')   # y축 숨기기
        self.plotWidget_2.getPlotItem().hideAxis('bottom') # x축 숨기기
        self.plotWidget_2.setYRange(-0.8, 0.8)
        self.plotWidget_2.setXRange(0, 200)
        self.plotWidget_2.enableAutoRange(axis='xy', enable=False)

        # 타이머 설정
        self.timer_1 = QTimer()
        self.timer_1.timeout.connect(self.update_plot_radar)
        self.timer_1.start(43)  # 50ms마다 업데이트
        self.heart_rate_graph.addWidget(self.plotWidget_1)
        self.resp_rate_graph.addWidget(self.plotWidget_2)

        # conductivity graph setting
        self.plotWidget_3 = pg.PlotWidget()
        self.plotWidget_3.setBackground('w')  # 배경색을 흰색으로 설정

        # 그래프 초기화
        self.plot_data = self.plotWidget_3.plot(self.xdata, self.ydata, pen=pg.mkPen('black', width=3))
        self.plot_data_2 = self.plotWidget_3.plot(self.xdata, self.ydata2, pen=pg.mkPen('purple', width=3))

        # x축과 y축 숨기기
        self.plotWidget_3.getPlotItem().hideAxis('left')   # y축 숨기기
        self.plotWidget_3.getPlotItem().hideAxis('bottom') # x축 숨기기
        self.plotWidget_3.setYRange(0, 100000)
        self.plotWidget_3.enableAutoRange(axis='xy', enable=False)
        
        # 타이머 설정
        self.timer_2 = QTimer()
        self.timer_2.timeout.connect(self.update_plot_conductivity)
        self.timer_2.start(40)  # 100ms마다 업데이트
        self.conductivity_graph.addWidget(self.plotWidget_3)

        self.timer_3=QTimer()
        self.timer_3.timeout.connect(self.lcd_number_update)
        self.timer_3.start(200)  # 100ms마다 업데이트

        self.timer_4=QTimer()
        self.timer_4.timeout.connect(self.update_image)
        self.timer_4.start(50)  # 100ms마다 업데이트
        # 이미지
        # presence 이미지 설정
        self.set_image_to_label(self.presence_image, "asset/presense_black.png")

        # drowsiness 이미지 설정
        self.set_image_to_label(self.drowsiness_image, "asset/drowsiness_red.png")

        # grip 이미지 설정
        self.set_image_to_label(self.grip_image, "asset/grip_black.svg")
        
        # warning 이미지 설정
        self.set_image_to_label(self.warning_image, "asset/warning_black.svg")
        
        #GIF 
        # Heart_rate GIF 설정
        self.setup_gif(self.heart_rate_gif, "asset/heart_rate.gif", self.update_heart_rate_frame)

        # Repo_rate GIF 설정
        self.setup_gif(self.resp_rate_gif, "asset/breath_rate.gif", self.update_repo_rate_frame)
    
    def send_command(self, command):
        self.feedback.write(command.encode('utf-8'))
        
    def set_progressbar_wheel(self, value):
        self.conductivity_progressbar.setValue(value)

    def set_image_to_label(self, label, image_path):
        image = QPixmap(image_path)
        # QLabel에 QPixmap 설정
        label.setPixmap(image)
    
    def setup_gif(self, graphics_view, gif_path, update_callback):
        # QGraphicsScene 설정
        scene = QGraphicsScene(self)
        graphics_view.setScene(scene)
        pixmap_item = QGraphicsPixmapItem()
        scene.addItem(pixmap_item)

        # QMovie 설정
        movie = QMovie(gif_path)
        movie.frameChanged.connect(lambda: update_callback(movie, pixmap_item, scene, graphics_view))
        movie.start()

    def update_heart_rate_frame(self, movie, pixmap_item, scene, graphics_view):
        pixmap = movie.currentPixmap()
        pixmap_item.setPixmap(pixmap)
        scene.setSceneRect(pixmap.rect())
        graphics_view.fitInView(scene.sceneRect(), Qt.IgnoreAspectRatio)

    def update_repo_rate_frame(self, movie, pixmap_item, scene, graphics_view):
        pixmap = movie.currentPixmap()
        pixmap_item.setPixmap(pixmap)
        scene.setSceneRect(pixmap.rect())
        graphics_view.fitInView(scene.sceneRect(), Qt.IgnoreAspectRatio)

    def update_plot_conductivity(self):
        self.plotWidget_3.setUpdatesEnabled(False)
        xdata_wheel, sensor_wheel, shield_wheel, conductivity, prediction = self.queue_wheel.get()
        self.set_progressbar_wheel(conductivity)
        
        self.grip = prediction
        if self.presence == 1:
            self.grip_queue.append(self.grip)
        else:
            self.grip_queue.append(1)    
            
        # 새로운 데이터를 추가 (움직이는 효과)
        self.ydata.append(sensor_wheel)
        self.ydata2.append(shield_wheel)

        # 기존 데이터를 삭제
        self.ydata = self.ydata[-200:]
        self.ydata2 = self.ydata2[-200:]

        self.plot_data.setData(self.xdata, self.ydata)
        self.plot_data_2.setData(self.xdata, self.ydata2)
        self.plotWidget_3.setUpdatesEnabled(True)

    def update_plot_radar(self):
        
        x_data, breathdata, heartdata, diff = self.queue.get()
        diff = 1
        
        if diff == 1:
            self.set_image_to_label(self.presence_image, "asset/presence_green.svg")
            self.presence = 1
            self.ydata_radar.append(heartdata)
            self.ydata_radar_2.append(breathdata)
            
        else:
            self.ydata_radar.append(0)
            self.ydata_radar_2.append(0)
            self.set_image_to_label(self.presence_image, "asset/presence_black.svg")
            self.set_image_to_label(self.drowsiness_image, "asset/drowsiness_black.png")
            self.set_image_to_label(self.grip_image, "asset/handle_black.svg")
            self.set_image_to_label(self.warning_image, "asset/warning_black.svg")        
            self.presence = 0

        self.ydata_radar = self.ydata_radar[-200:]
        self.ydata_radar_2 = self.ydata_radar_2[-200:]
        
        self.plot1.setData(self.xdata_radar, self.ydata_radar)
        self.plot2.setData(self.xdata_radar, self.ydata_radar_2)
    
    def update_image(self):
        if self.presence == 1:
            # Grip Update
            if np.sum(np.array(self.grip_queue)[:]) < 20 and np.sum(np.array(self.grip_queue)[100:]) >= 10:            
                self.set_image_to_label(self.grip_image, "asset/handle_green.svg")
                if self.grip_stat != 0:
                    self.send_command('Z')
                    self.grip_stat = 0
            elif np.sum(np.array(self.grip_queue)[:]) < 20:            
                self.set_image_to_label(self.grip_image, "asset/handle_red.svg")
                if self.grip_stat != 2:
                    self.send_command('B')
                    self.grip_stat = 2
            elif np.sum(np.array(self.grip_queue)[100:]) < 10: 
                self.set_image_to_label(self.grip_image, "asset/handle_orange.svg")
                if self.grip_stat != 1:
                    self.send_command('A')
                    self.grip_stat = 1
            else: 
                self.set_image_to_label(self.grip_image, "asset/handle_green.svg")
                if self.grip_stat != 0:
                    self.send_command('Z')
                    self.grip_stat = 0
            
        if np.sum(np.array(self.sleep_stage_short)[:]) == 2:
            if np.sum(np.array(self.sleep_stage_long)[:]) == 5:
                self.drowsiness_state = 2
            else:
                self.drowsiness_state = 1
        else: 
            self.drowsiness_state = 0
        
        if self.presence == 1:
            # Drowsiness Update
            if self.drowsiness_state == 0 :  ## normal state
                if self.drowsiness_stat != 0:
                    self.send_command('Z')
                    self.set_image_to_label(self.drowsiness_image, "asset/drowsiness_green.png")
                    self.set_image_to_label(self.warning_image, "asset/warning_green.svg")
                    self.drowsiness_stat = 0
            if self.drowsiness_state == 1 and self.grip == 1: ## half drowsiness + safety_grip
                if self.drowsiness_stat != 1:
                    self.set_image_to_label(self.drowsiness_image, "asset/drowsiness_orange.png")
                    self.send_command('C')
                    self.drowsiness_stat = 1
                # print('C')
            if self.drowsiness_state == 1 and self.grip == 0:  ## half drowsiness + unsafety_grip
                if self.drowsiness_stat != 2:
                    self.set_image_to_label(self.drowsiness_image, "asset/drowsiness_orange.png")
                    self.set_image_to_label(self.warning_image, "asset/warning_orange.svg")
                    self.send_command('E')
                    self.drowsiness_stat = 2
                # print('E')
            if self.drowsiness_state == 2 and self.grip == 1:  ## drowsiness + safety_grip
                if self.drowsiness_stat != 3:
                    self.set_image_to_label(self.drowsiness_image, "asset/drowsiness_red.png")
                    self.send_command('D')
                    self.drowsiness_stat = 3
                # print('D')

            if self.drowsiness_state == 2 and self.grip == 0: ## drowsiness + unsafety_grip
                if self.drowsiness_stat != 4:
                    self.set_image_to_label(self.drowsiness_image, "asset/drowsiness_red.png")
                    self.set_image_to_label(self.warning_image, "asset/warning_red.svg")
                    self.send_command('F')
                    self.drowsiness_stat = 4
                # print('F')
                
    def lcd_number_update(self):
        if not self.queue_AI.empty():
            hr, rr, sleep_stage, drowsiness = queue_AI.get()
            self.drowsiness = drowsiness
            self.sleep_stage_short.append(drowsiness)
            self.sleep_stage_long.append(drowsiness)
            
            self.heart_rate_number.display(hr)
            self.resp_rate_number.display(rr)
        if self.presence == 0:
            self.heart_rate_number.display(0)
            self.resp_rate_number.display(0)
        
    
def start_thread(queue):
    # 데이터 수집 프로세스를 시작합니다.
    threading.Thread(target=process_data, args=(queue,), daemon=True).start()

def start_thread_wheel(queue):
    # 데이터 수집 프로세스를 시작합니다.
    threading.Thread(target=process_data_wheel, args=(queue,), daemon=True).start()
    
def start_thread_ai(queue):
    threading.Thread(target=process_AI, args=(queue,), daemon=True).start()


if __name__ == "__main__":
    queue = Queue(maxsize=1)
    queue_wheel = Queue()
    queue_AI = Queue()
    
    uart_thread = threading.Thread(target=read_uart, daemon=True)
    uart_thread.start()

    uart_thread_wheel = threading.Thread(target=read_uart_wheel, daemon=True)
    uart_thread_wheel.start()

    start_thread(queue)
    start_thread_wheel(queue_wheel)
    start_thread_ai(queue_AI)

    app = QApplication(sys.argv)
    window = MainWindow(queue, queue_wheel, queue_AI)
    
    window.show()
    sys.exit(app.exec())

