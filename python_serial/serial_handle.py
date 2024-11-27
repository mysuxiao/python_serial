import serial
import numpy as np
from scipy import stats
import pywt
from collections import deque

class SerialHandler:
    def __init__(self, port='COM3', baudrate=115200):
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            print(f"串口初始化成功")
            self.buffer = ""

            # 初始化参数
            self.warmup_samples = 100
            self.is_warmed_up = False
            self.warmup_buffer = [[] for _ in range(12)]
            self.baselines = [0] * 12
            self.scaling_factor = 10000

            # 小波变换参数
            self.wavelet_type = 'db4'  # 德拜小波
            self.wavelet_level = 3  # 分解层数
            self.signal_buffers = [deque(maxlen=64) for _ in range(12)]  # 存储信号用于小波变换

            # 采样率相关参数
            self.fs = 250  # 采样频率 Hz
            self.sample_period = 1000 / self.fs  # 采样周期（毫秒）
            self.sample_count = 0  # 采样点计数器

            # R波检测相关参数
            self.data_buffer = deque(maxlen=3)  # ECG数据缓冲区
            self.threshold = 0.4  # R波检测阈值
            self.last_r_peak_sample = None  # 上一个R波峰值的采样点编号
            self.rr_intervals = []  # 存储RR间期（单位：毫秒）

        except Exception as e:
            print(f"串口初始化失败: {e}")
            self.serial_port = None

    def wavelet_denoise(self, data):
        """使用小波变换进行去噪"""
        try:
            # 进行小波分解
            coeffs = pywt.wavedec(data, self.wavelet_type, level=self.wavelet_level)

            # 计算阈值
            threshold = np.median(np.abs(coeffs[-1])) * 1.4826 * np.sqrt(2 * np.log(len(data)))

            def safe_threshold(c, threshold):
                with np.errstate(divide='ignore', invalid='ignore'):
                    thresholded = pywt.threshold(c, threshold, mode='soft')

                return np.nan_to_num(thresholded, nan=0.0, posinf=0.0, neginf=0.0)

            coeffs_thresholded = [safe_threshold(c, threshold) for c in coeffs]

            # 重构信号
            denoised = pywt.waverec(coeffs_thresholded, self.wavelet_type)

            # 确保输出长度与输入相同并处理可能的 nan 值
            result = denoised[:len(data)]
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            # 添加新的滤除条件：将大于0.9或小于-1的点置0
            result[result > 0.7] = 0
            result[result < -0.7] = 0

            return result

        except Exception as e:
            print(f"小波去噪错误: {e}")
            return np.zeros_like(data)

    def process_value(self, value):
        """保留符号并只取后四位数据"""
        sign = 1 if value >= 0 else -1
        abs_value = abs(value)
        return sign * (abs_value % 100000)

    def normalize_value(self, value, channel_index):
        """基于基线的归一化，并应用小波去噪"""
        try:
            if not self.is_warmed_up:
                self.warmup_buffer[channel_index].append(value)
                if len(self.warmup_buffer[channel_index]) >= self.warmup_samples:
                    self.baselines[channel_index] = np.mean(self.warmup_buffer[channel_index])

                    if all(len(buffer) >= self.warmup_samples for buffer in self.warmup_buffer):
                        self.is_warmed_up = True
                        print("预热完成，开始正常数据采集")
                return 0

            normalized = (value - self.baselines[channel_index]) / self.scaling_factor
            normalized = max(min(normalized, 1), -1)

            # 将归一化后的值添加到缓冲区
            self.signal_buffers[channel_index].append(normalized)

            # 当收集够足够的数据点时进行小波去噪
            if len(self.signal_buffers[channel_index]) == self.signal_buffers[channel_index].maxlen:
                signal_array = np.array(self.signal_buffers[channel_index])
                denoised_signal = self.wavelet_denoise(signal_array)
                result = denoised_signal[-1]  # 获取最新的去噪后的值

                # 对去噪后的信号也进行检查
                if abs(result) == 1:
                    result = 0

                # 对ECG导联的数据进行R波检测
                if channel_index == 0:
                    self.detect_r_peak(result)

                return result

            return normalized

        except Exception as e:
            print(f"归一化错误: {e}")
            return 0

    def detect_r_peak(self, value):
        """使用采样点计数的R波检测算法"""
        try:
            self.data_buffer.append((self.sample_count, value))

            if len(self.data_buffer) < 3:
                return

            # 检查中间点是否是局部最大值且超过阈值
            _, prev_val = self.data_buffer[-3]
            current_sample, current_val = self.data_buffer[-2]
            _, next_val = self.data_buffer[-1]

            if (current_val > prev_val and
                    current_val > next_val and
                    current_val > self.threshold):

                if self.last_r_peak_sample is None:
                    self.last_r_peak_sample = current_sample
                else:
                    # 计算RR间期（毫秒）
                    samples_between = current_sample - self.last_r_peak_sample
                    rr_interval = samples_between * self.sample_period

                    # 更新RR间期
                    self.update_rr_intervals(rr_interval)
                    self.last_r_peak_sample = current_sample

        except Exception as e:
            print(f"R波检测错误: {e}")

    def update_rr_intervals(self, new_rr_interval):
        """更新RR间期列表"""
        self.rr_intervals.append(new_rr_interval)
        # 保持列表长度在一个合理范围内
        if len(self.rr_intervals) > 100:  # 保留最近100个RR间期
            self.rr_intervals.pop(0)

    def get_hrv_data(self):
        """获取HRV相关数据"""
        try:
            if len(self.rr_intervals) < 2:
                return {
                    'heart_rate': 0,
                    'SDNN': 0,
                    'RMSSD': 0,
                    'pNN50': 0
                }

            # 计算心率 (bpm)
            current_hr = (30000*6/7) / self.rr_intervals[-1] if self.rr_intervals else 0

            # 使用最近的N个RR间期进行HRV分析
            n_intervals = min(len(self.rr_intervals), 10)
            recent_rr = np.array(self.rr_intervals[-n_intervals:])

            # 计算SDNN
            sdnn = np.std(recent_rr) if len(recent_rr) > 1 else 0

            # 计算RMSSD
            rr_diff = np.diff(recent_rr)
            rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else 0

            # 计算pNN50
            nn50 = np.sum(np.abs(rr_diff) > 50)
            pnn50 = (nn50 / len(rr_diff) * 100) if len(rr_diff) > 0 else 0

            return {
                'heart_rate': current_hr,
                'SDNN': sdnn,
                'RMSSD': rmssd,
                'pNN50': pnn50
            }
        except Exception as e:
            print(f"计算HRV指标时出错: {str(e)}")
            return {
                'heart_rate': 0,
                'SDNN': 0,
                'RMSSD': 0,
                'pNN50': 0
            }

    def read_data(self):
        if not self.serial_port:
            return None

        try:
            while self.serial_port.in_waiting:
                self.buffer += self.serial_port.read().decode('ascii')

            if '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                line = line.strip()

                if not line:
                    return None

                str_values = [val.strip() for val in line.split(';') if val.strip()]

                if len(str_values) != 12:
                    return None

                try:
                    values = [int(val) for val in str_values]
                    processed_values = [self.process_value(val) for val in values]
                    normalized_values = [self.normalize_value(val, i) for i, val in enumerate(processed_values)]

                    if not self.is_warmed_up:
                        return None

                    # 增加采样点计数
                    self.sample_count += 1
                    return normalized_values

                except ValueError:
                    print("数据转换错误")
                    return None

        except Exception as e:
            print(f"数据读取错误: {e}")
            self.buffer = ""
            return None

        return None

    def close(self):
        """关闭串口连接"""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("串口已关闭")