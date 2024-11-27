from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QLabel, QSpinBox, QStackedWidget,
                             QGroupBox, QFormLayout, QGridLayout, QFrame)
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg
import numpy as np
import serial.tools.list_ports
from collections import deque


class DataReceiver(QObject):
    hrv_data_updated = pyqtSignal(dict)


class ECGMonitorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_plots()
        from PyQt5.QtGui import QIcon
        self.setWindowIcon(QIcon('icon.ico'))
        self.data_receiver = DataReceiver()
        self.data_receiver.hrv_data_updated.connect(self.update_hrv_display)

        self.data_buffers = [deque(maxlen=self.buffer_size) for _ in range(12)]
        for buffer in self.data_buffers:
            buffer.extend([0] * self.buffer_size)

    def setup_ui(self):
        """设置基本UI元素"""
        self.setWindowTitle("12导联心电实时监护系统")
        self.resize(1600, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                margin-top: 1ex;
                font-size: 14px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 5px;
            }
            QLabel {
                color: #ffffff;
                font-size: 13px;
            }
            QPushButton {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px 15px;
                min-width: 80px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                min-width: 100px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # 顶部控制面板
        top_panel = QWidget()
        top_panel.setMaximumHeight(200)  # 增加顶部面板高度
        top_layout = QHBoxLayout(top_panel)
        top_layout.setSpacing(20)
        top_layout.setContentsMargins(5, 5, 5, 5)

        # 串口设置组
        serial_group = QGroupBox("串口设置")
        serial_layout = QHBoxLayout()
        serial_layout.setSpacing(15)

        port_layout = QVBoxLayout()
        port_label = QLabel("串口:")
        port_label.setStyleSheet("font-size: 14px;")
        port_layout.addWidget(port_label)
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(120)
        port_layout.addWidget(self.port_combo)

        baud_layout = QVBoxLayout()
        baud_label = QLabel("波特率:")
        baud_label.setStyleSheet("font-size: 14px;")
        baud_layout.addWidget(baud_label)
        self.baudrate_combo = QComboBox()
        standard_baudrates = ['9600', '19200', '38400', '57600', '115200']
        self.baudrate_combo.addItems(standard_baudrates)
        self.baudrate_combo.setCurrentText('115200')
        baud_layout.addWidget(self.baudrate_combo)

        self.refresh_ports_button = QPushButton("刷新串口")
        self.refresh_ports_button.setIcon(self.style().standardIcon(self.style().SP_BrowserReload))

        serial_layout.addLayout(port_layout)
        serial_layout.addLayout(baud_layout)
        serial_layout.addWidget(self.refresh_ports_button)
        serial_group.setLayout(serial_layout)

        # 数据显示组
        data_group = QGroupBox("实时数据监测")
        data_layout = QHBoxLayout()
        data_layout.setSpacing(20)  # 增加间距

        self.data_labels = {}

        # 创建数值显示面板
        # 创建数值显示面板
        value_panels = [
            ('heart_rate', "heart rate", "bpm", "#ff4444"),  # 红色
            ('SDNN', "SDNN", "ms", "#33ff33"),  # 绿色
            ('RMSSD', "RMSSD", "ms", "#33ff33"),  # 绿色
            ('pNN50', "PNN50", "%", "#33ff33")  # 绿色
        ]

        for key, name, unit, color in value_panels:
            panel = QFrame()
            panel.setMinimumWidth(180)  # 增加最小宽度
            panel.setMinimumHeight(160)  # 增加最小高度
            panel.setStyleSheet(f"""
                    QFrame {{
                        background-color: #3d3d3d;
                        border-radius: 8px;
                        padding: 10px;
                    }}
                """)
            panel_layout = QVBoxLayout(panel)
            panel_layout.setSpacing(8)

            name_label = QLabel(f"{name}")
            name_label.setStyleSheet(f"""
                    font-size: 20px;
                    color: {color};
                    font-weight: bold;
                """)
            name_label.setAlignment(Qt.AlignCenter)

            value_label = QLabel("0")
            value_label.setStyleSheet(f"""
                    font-size: 18px;
                    color: {color};
                    font-weight: bold;
                """)
            value_label.setAlignment(Qt.AlignCenter)

            unit_label = QLabel(f"({unit})")
            unit_label.setStyleSheet(f"""
                    font-size: 18px;
                    color: {color};
                """)
            unit_label.setAlignment(Qt.AlignCenter)

            panel_layout.addWidget(name_label)
            panel_layout.addWidget(value_label)
            panel_layout.addWidget(unit_label)

            self.data_labels[key] = value_label
            data_layout.addWidget(panel)

        data_group.setLayout(data_layout)

        # 控制按钮组
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        control_layout.setSpacing(15)

        self.start_button = QPushButton("开始采集")
        self.stop_button = QPushButton("停止采集")
        self.switch_view_button = QPushButton("切换视图")
        self.clear_button = QPushButton("清除数据")

        # 设置按钮大小
        for button in [self.start_button, self.stop_button, self.switch_view_button, self.clear_button]:
            button.setMinimumWidth(100)
            button.setMinimumHeight(35)
            button.setStyleSheet("""
                QPushButton {
                    font-size: 14px;
                    padding: 5px 15px;
                }
            """)
            control_layout.addWidget(button)

        self.stop_button.setEnabled(False)
        control_group.setLayout(control_layout)

        # 添加所有组件到顶部面板
        top_layout.addWidget(serial_group)
        top_layout.addWidget(data_group)
        top_layout.addWidget(control_group)
        top_layout.addStretch()

        self.main_layout.addWidget(top_panel)

        # 图表显示区域
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setStyleSheet("""
            QStackedWidget {
                background-color: #1e1e1e;
                border: 2px solid #4a4a4a;
                border-radius: 6px;
            }
        """)

        self.limb_leads_widget = pg.GraphicsLayoutWidget()
        self.chest_leads_widget = pg.GraphicsLayoutWidget()

        self.stacked_widget.addWidget(self.limb_leads_widget)
        self.stacked_widget.addWidget(self.chest_leads_widget)

        self.main_layout.addWidget(self.stacked_widget, stretch=5)

        self.update_port_list()
        self.refresh_ports_button.clicked.connect(self.update_port_list)
    def setup_plots(self):
        """设置图表"""
        self.sample_rate = 360
        self.time_window = 2
        self.buffer_size = int(self.time_window * self.sample_rate)
        self.time_array = np.linspace(0, self.time_window, self.buffer_size)

        # 设置绘图样式
        pg.setConfigOptions(antialias=True)

        # 设置背景颜色
        self.limb_leads_widget.setBackground('#1e1e1e')
        self.chest_leads_widget.setBackground('#1e1e1e')

        self.plots = []
        self.curves = []

        plot_params = {
            'limb': {
                'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'],
                'widget': self.limb_leads_widget
            },
            'chest': {
                'leads': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                'widget': self.chest_leads_widget
            }
        }

        for plot_type in plot_params.values():
            for i, lead in enumerate(plot_type['leads']):
                plot = plot_type['widget'].addPlot(row=i // 2, col=i % 2)
                self._setup_single_plot(plot, lead)
                curve = plot.plot(pen=pg.mkPen('#00ff00', width=1.5))
                self.plots.append(plot)
                self.curves.append(curve)

        self.switch_view_button.clicked.connect(self.switch_view)
        self.clear_button.clicked.connect(self.clear_plots)

    def _setup_single_plot(self, plot, lead_name):
        """配置单个图表的通用属性"""
        plot.setTitle(f'导联 {lead_name}', color='#ffffff', size='14pt')
        plot.setLabel('left', '振幅', color='#ffffff')

        plot.getAxis('bottom').hide()
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setXRange(0, self.time_window, padding=0)
        plot.setYRange(-1.5, 1.5, padding=0)

        # 设置网格样式
        plot.getAxis('left').setPen(pg.mkPen(color='#ffffff', width=1))
        plot.getAxis('bottom').setPen(pg.mkPen(color='#ffffff', width=1))

        plot.setMouseEnabled(x=False, y=False)
        plot.hideButtons()
        plot.setMenuEnabled(False)

    def update_port_list(self):
        """更新可用串口列表"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)

    def update_hrv_display(self, hrv_dict):
        """更新HRV指标显示"""
        # 更新显示
        if 'heart_rate' in hrv_dict:
            self.data_labels['heart_rate'].setText(f"{hrv_dict['heart_rate']:.0f}")
        for key in ['SDNN', 'RMSSD', 'pNN50']:
            if key in hrv_dict and key in self.data_labels:
                self.data_labels[key].setText(f"{hrv_dict[key]:.1f}")

    def update_plot_data(self, channel, value):
        """更新指定通道的数据"""
        if 0 <= channel < len(self.data_buffers):
            self.data_buffers[channel].append(value)
            self.curves[channel].setData(self.time_array, list(self.data_buffers[channel]))

    def switch_view(self):
        """切换视图"""
        current_index = self.stacked_widget.currentIndex()
        new_index = (current_index + 1) % 2
        self.stacked_widget.setCurrentIndex(new_index)

    def clear_plots(self):
        """清除所有图表数据"""
        # 重置所有数据缓冲区
        for i, buffer in enumerate(self.data_buffers):
            buffer.clear()
            buffer.extend([0] * self.buffer_size)
            # 更新曲线显示
            self.curves[i].setData(self.time_array, list(buffer))

        # 重置所有数值显示
        self.data_labels['heart_rate'].setText("0")
        for key in ['SDNN', 'RMSSD', 'pNN50']:
            self.data_labels[key].setText("0")