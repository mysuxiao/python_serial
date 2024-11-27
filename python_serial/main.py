import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from serial_handle import SerialHandler
from ui import ECGMonitorUI


class ECGController:
    def __init__(self):
        self.ui = ECGMonitorUI()
        self.serial_handler = None

        # 连接UI控件信号
        self.ui.start_button.clicked.connect(self.start_acquisition)
        self.ui.stop_button.clicked.connect(self.stop_acquisition)

        # 创建定时器用于定期更新数据
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)

    def start_acquisition(self):
        port = self.ui.port_combo.currentText()
        baudrate = int(self.ui.baudrate_combo.currentText())

        self.serial_handler = SerialHandler(port=port, baudrate=baudrate)
        self.update_timer.start(20)  # 50Hz更新率

        self.ui.start_button.setEnabled(False)
        self.ui.stop_button.setEnabled(True)

    def stop_acquisition(self):
        if self.serial_handler:
            self.serial_handler.close()
            self.serial_handler = None
        self.update_timer.stop()

        self.ui.start_button.setEnabled(True)
        self.ui.stop_button.setEnabled(False)
        self.ui.clear_plots()

    def update_data(self):
        if not self.serial_handler:
            return

        # 读取数据
        data = self.serial_handler.read_data()
        if data is not None:
            # 更新波形图
            for i, value in enumerate(data):
                self.ui.update_plot_data(i, value)

            # 获取并更新HRV数据
            hrv_data = self.serial_handler.get_hrv_data()
            self.ui.data_receiver.hrv_data_updated.emit(hrv_data)

    def show(self):
        self.ui.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion主题以获得更好的外观

    controller = ECGController()
    controller.show()

    sys.exit(app.exec_())