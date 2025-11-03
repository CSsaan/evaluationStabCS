import sys
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QDoubleSpinBox, QGroupBox, QTextEdit, QMessageBox, QProgressBar)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# 工作线程类，用于在后台执行耗时操作
class EvaluationWorker(QThread):
    # 定义信号，用于向主线程传递结果
    finished = pyqtSignal(object, object)  # (metrics_result, trajectory_data)
    error = pyqtSignal(str)  # 错误信息
    
    def __init__(self, original_video_path, pred_video_path, scale_factor):
        super().__init__()
        self.original_video_path = original_video_path
        self.pred_video_path = pred_video_path
        self.scale_factor = scale_factor
    
    def run(self):
        try:
            from src.metrics_video_faster import metrics
            from src.trajectory import gen_trajectory_data
            
            # 执行评估
            CR_AVG_MIN, DVDV, SS_AVG_T_R = metrics(self.original_video_path, self.pred_video_path, scale_factor=self.scale_factor, visualize=True)
            # 生成轨迹
            trajectory1, trajectory2 = gen_trajectory_data(self.original_video_path, self.pred_video_path, self.scale_factor)
            
            # 将轨迹数据发送到主线程，让主线程负责绘图
            trajectory_data = {
                'trajectory1': trajectory1,
                'trajectory2': trajectory2
            }
            
            # 发送结果到主线程
            self.finished.emit((CR_AVG_MIN, DVDV, SS_AVG_T_R), trajectory_data)
        except Exception as e:
            # 发送错误信息到主线程
            self.error.emit(str(e))


class VideoStabilityUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None  # 工作线程实例
        self.trajectory_canvas = None  # 用于显示轨迹图的canvas
        self.trajectory_data = None  # 保存轨迹数据用于保存图片
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Video Stabilization Evaluator')
        self.setGeometry(100, 100, 1000, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 右侧可视化区域
        self.create_visualization_area(main_layout)
        
    def is_dark_mode(self):
        """
        检查系统是否处于深色模式
        """
        palette = self.palette()
        window_color = palette.color(QPalette.ColorRole.Window)
        # 简单判断：如果背景颜色较暗，则认为是深色模式
        return window_color.lightness() < 128
    
    def create_visualization_area(self, parent_layout):
        # 创建一个空的figure用于占位
        if self.is_dark_mode(): # 深色样式
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        parent_layout.addWidget(self.canvas, 3)
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout(file_group)
        
        self.original_path_label = QLabel("原始视频: 未选择")
        self.pred_path_label = QLabel("稳定视频: 未选择")
        
        orig_btn = QPushButton("选择原始视频")
        pred_btn = QPushButton("选择稳定视频")
        
        orig_btn.clicked.connect(self.select_original_video)
        pred_btn.clicked.connect(self.select_pred_video)
        
        file_layout.addWidget(self.original_path_label)
        file_layout.addWidget(orig_btn)
        file_layout.addWidget(self.pred_path_label)
        file_layout.addWidget(pred_btn)
        
        # 参数设置组
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout(param_group)
        
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("缩放因子:"))
        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setRange(0.1, 2.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(1.0)
        scale_layout.addWidget(self.scale_spinbox)
        param_layout.addLayout(scale_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        param_layout.addWidget(self.progress_bar)
        
        # 运行按钮
        self.run_btn = QPushButton("运行评估")
        self.run_btn.clicked.connect(self.run_evaluation)
        param_layout.addWidget(self.run_btn)
        
        # 图片保存按钮
        self.save_image_btn = QPushButton("保存轨迹图")
        self.save_image_btn.clicked.connect(self.save_trajectory_image)
        self.save_image_btn.setEnabled(False)  # 初始禁用，只有在有图像时才启用
        param_layout.addWidget(self.save_image_btn)
        
        # 结果显示
        result_group = QGroupBox("评估结果")
        result_layout = QVBoxLayout(result_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        
        layout.addWidget(file_group)
        layout.addWidget(param_group)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.save_image_btn)
        layout.addWidget(result_group)
        layout.addStretch()
        
        return panel
        
    def select_original_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择原始视频", "", "Video Files (*.avi *.mp4 *.mov)")
        if file_path:
            self.original_path_label.setText(f"原始视频: {file_path}")
            self.original_video_path = file_path
            
    def select_pred_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择稳定视频", "", "Video Files (*.avi *.mp4 *.mov)")
        if file_path:
            self.pred_path_label.setText(f"稳定视频: {file_path}")
            self.pred_video_path = file_path
            
    def run_evaluation(self):
        # 检查是否已选择文件
        if not hasattr(self, 'original_video_path') or not hasattr(self, 'pred_video_path'):
            QMessageBox.warning(self, "警告", "请先选择原始视频和稳定视频文件")
            return
            
        # 禁用运行按钮，显示进度条
        self.run_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)  # 禁用保存按钮
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 设置为不确定模式（忙碌指示器）
        self.result_text.setText("正在处理中，请稍候...")
        
        # 创建并启动工作线程
        scale_factor = self.scale_spinbox.value()
        self.worker = EvaluationWorker(
            self.original_video_path, 
            self.pred_video_path, 
            scale_factor
        )
        
        # 连接信号和槽
        self.worker.finished.connect(self.on_evaluation_finished)
        self.worker.error.connect(self.on_evaluation_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        
        # 启动线程
        self.worker.start()
            
    def on_evaluation_finished(self, metrics_result, trajectory_data):
        # 启用运行按钮，隐藏进度条
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # 保存轨迹数据
        self.trajectory_data = trajectory_data
        
        # 解包结果
        CR_AVG_MIN, DVDV, SS_AVG_T_R = metrics_result
        # 显示结果
        result_str = f"CroppingRatio(average, min) ↑:  {CR_AVG_MIN[0]:.4f}, {CR_AVG_MIN[1]:.4f}\n"
        result_str += f"DirectionalVariation ↑: {DVDV:.4f}\n"
        result_str += f"StabilityScore(average, trans, rotate) ↑: {SS_AVG_T_R[0]:.4f}, {SS_AVG_T_R[1]:.4f}, {SS_AVG_T_R[2]:.4f}"
        self.result_text.setText(result_str)
        
        # 更新轨迹图
        from src.trajectory import plot_trajectory_data
        plot_trajectory_data(self.figure, trajectory_data)
        # 重新绘制canvas
        self.figure.tight_layout()
        self.canvas.draw()
        
        # 启用保存按钮
        self.save_image_btn.setEnabled(True)
        
        # 显示完成消息
        QMessageBox.information(self, "完成", "视频稳定性评估已完成")
        
    def on_evaluation_error(self, error_message):
        # 启用运行按钮，隐藏进度条
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # 显示错误信息
        self.result_text.setText(f"评估出错: {error_message}")
        QMessageBox.critical(self, "错误", f"评估过程中发生错误:\n{error_message}")

    def save_trajectory_image(self):
        """
        保存轨迹图为图片文件
        """
        if self.trajectory_data is None:
            QMessageBox.warning(self, "警告", "没有可保存的轨迹图")
            return
            
        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存轨迹图", "trajectory.png", 
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;PDF 文件 (*.pdf);;SVG 文件 (*.svg)")

        if file_path:
            try:
                # 保存当前图表
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "成功", f"轨迹图已保存至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图片时出错:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = VideoStabilityUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
