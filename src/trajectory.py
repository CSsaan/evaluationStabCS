from .vidstab import VidStab
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def gen_trajectory_data(video1_path, video2_path, resolution_option='native'):
    # 生成两个视频的轨迹数据
    stabilizer1 = VidStab(resolution_option=resolution_option)
    stabilizer1.gen_transforms(input_path=video1_path)
    _transforms1, trajectory1 = stabilizer1.get_transforms_trajectory() # (N, 3)
    all_transforms1 = stabilizer1.get_all_transforms()

    stabilizer2 = VidStab(resolution_option=resolution_option)
    stabilizer2.gen_transforms(input_path=video2_path)
    _transforms2, trajectory2 = stabilizer2.get_transforms_trajectory()
    all_transforms2 = stabilizer2.get_all_transforms()

    return trajectory1, trajectory2, all_transforms1, all_transforms2

def plot_trajectory_data(figure, trajectory_data):
    # 清除当前图形
    figure = figure
    figure.clear()
    
    trajectory1 = trajectory_data['trajectory1']
    trajectory2 = trajectory_data['trajectory2']
    all_transforms1 = trajectory_data['all_transforms1']
    all_transforms2 = trajectory_data['all_transforms2']
    
    # 处理不同长度的轨迹数据
    min_length = min(trajectory1.shape[0], trajectory2.shape[0])
    
    with plt.style.context('ggplot'):
        # 创建子图
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212, sharex=ax1)

        # x方向双y轴显示
        # 左侧y轴显示原始轨迹
        ax1_left = ax1
        line1 = ax1_left.plot(trajectory1[:min_length, 0], label='Original dx', color='C0')
        ax1_left.tick_params(axis='y', labelcolor='C0')
        
        # 右侧y轴显示稳定后轨迹
        ax1_right = ax1.twinx()
        line2 = ax1_right.plot(trajectory2[:min_length, 0], label='Stabilized dx', color='C1')
        ax1_right.tick_params(axis='y', labelcolor='C1')

        ax1_left.set_ylabel('Original dx', color='C0')
        ax1_right.set_ylabel('Stabilized dx', color='C1')
        
        # 合并图例
        lines1 = line1 + line2
        labels1 = [l.get_label() for l in lines1]
        ax1_left.legend(lines1, labels1, loc='upper left')

        # y方向双y轴显示
        # 左侧y轴显示原始轨迹
        ax2_left = ax2
        line3 = ax2_left.plot(trajectory1[:min_length, 1], label='Original dy', color='C2')
        ax2_left.tick_params(axis='y', labelcolor='C2')
        
        # 右侧y轴显示稳定后轨迹
        ax2_right = ax2.twinx()
        line4 = ax2_right.plot(trajectory2[:min_length, 1], label='Stabilized dy', color='C3')
        ax2_right.tick_params(axis='y', labelcolor='C3')

        ax2_left.set_ylabel('Original dy', color='C2')
        ax2_right.set_ylabel('Stabilized dy', color='C3')
        
        # 合并图例
        lines2 = line3 + line4
        labels2 = [l.get_label() for l in lines2]
        ax2_left.legend(lines2, labels2, loc='upper left')

        # 设置x轴标签
        ax2_left.set_xlabel('Frame Number')

        figure.suptitle('Video Trajectory Comparison', x=0.15, y=0.96, ha='left')
        figure.tight_layout()

    # 绘制all_transforms数据（新增功能）
    if all_transforms1 is not None and all_transforms2 is not None:
        # 确保数据长度一致
        min_transforms_length = min(len(all_transforms1), len(all_transforms2))
        
        if min_transforms_length > 0:
            # 提取dx, dy, rotation数据
            dx1 = [transform[0] for transform in all_transforms1[:min_transforms_length]]
            dy1 = [transform[1] for transform in all_transforms1[:min_transforms_length]]
            rot1 = [transform[2] for transform in all_transforms1[:min_transforms_length]]
            
            dx2 = [transform[0] for transform in all_transforms2[:min_transforms_length]]
            dy2 = [transform[1] for transform in all_transforms2[:min_transforms_length]]
            rot2 = [transform[2] for transform in all_transforms2[:min_transforms_length]]
            
            frame_indices = list(range(min_transforms_length))
            
            # 创建并保存dx对比图
            plt.ioff()  # 关闭交互模式
            fig_dx = plt.figure(figsize=(10, 6))
            plt.plot(frame_indices, dx1, label='Original dx', color='C0', linewidth=1.5)
            plt.plot(frame_indices, dx2, label='Stabilized dx', color='C1', linewidth=1.5)
            plt.xlabel('Frame Number')
            plt.ylabel('dx (pixels)')
            plt.title('X-axis Translation Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_dx.savefig('logs/dx_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig_dx)
            
            # 创建并保存dy对比图
            fig_dy = plt.figure(figsize=(10, 6))
            plt.plot(frame_indices, dy1, label='Original dy', color='C2', linewidth=1.5)
            plt.plot(frame_indices, dy2, label='Stabilized dy', color='C3', linewidth=1.5)
            plt.xlabel('Frame Number')
            plt.ylabel('dy (pixels)')
            plt.title('Y-axis Translation Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_dy.savefig('logs/dy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig_dy)
            
            # 创建并保存rotation对比图
            fig_rot = plt.figure(figsize=(10, 6))
            plt.plot(frame_indices, rot1, label='Original rotation', color='C4', linewidth=1.5)
            plt.plot(frame_indices, rot2, label='Stabilized rotation', color='C5', linewidth=1.5)
            plt.xlabel('Frame Number')
            plt.ylabel('Rotation (radians)')
            plt.title('Rotation Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_rot.savefig('logs/rotation_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig_rot)

# def plot_trajectory_comparison(video1_path, video2_path, scale_factor=1.0, output_path='./trajectory_plot1.png'):
#     """
#     比较两个视频的轨迹数据并生成双Y轴图表
    
#     :param video1_path: 第一个视频文件路径（通常是原始视频）
#     :param video2_path: 第二个视频文件路径（通常是稳定后的视频）
#     :param output_path: 输出图表文件路径
#     :return: tuple of matplotlib objects (Figure, (AxesSubplot, AxesSubplot))
#     """
    
#     # 生成两个视频的轨迹数据
#     stabilizer1 = VidStab(scale_factor=scale_factor)
#     stabilizer1.gen_transforms(input_path=video1_path)
#     transforms1, trajectory1 = stabilizer1.get_transforms_trajectory() # (N, 3)

#     stabilizer2 = VidStab(scale_factor=scale_factor)
#     stabilizer2.gen_transforms(input_path=video2_path)
#     transforms2, trajectory2 = stabilizer2.get_transforms_trajectory()

#     with plt.style.context('ggplot'):
#         fig, (ax1, ax2) = plt.subplots(2, sharex='all', figsize=(10, 8))

#         # 处理不同长度的轨迹数据
#         min_length = min(trajectory1.shape[0], trajectory2.shape[0])
        
#         # x方向双y轴显示
#         # 左侧y轴显示原始轨迹
#         ax1_left = ax1
#         line1 = ax1_left.plot(trajectory1[:min_length, 0], label='Original dx', color='C0')
#         ax1_left.set_ylabel('Original dx', color='C0')
#         ax1_left.tick_params(axis='y', labelcolor='C0')
        
#         # 右侧y轴显示稳定后轨迹
#         ax1_right = ax1.twinx()
#         line2 = ax1_right.plot(trajectory2[:min_length, 0], label='Stabilized dx', color='C1')
#         ax1_right.set_ylabel('Stabilized dx', color='C1')
#         ax1_right.tick_params(axis='y', labelcolor='C1')
        
#         # 合并图例
#         lines1 = line1 + line2
#         labels1 = [l.get_label() for l in lines1]
#         ax1_left.legend(lines1, labels1, loc='upper left')

#         # y方向双y轴显示
#         # 左侧y轴显示原始轨迹
#         ax2_left = ax2
#         line3 = ax2_left.plot(trajectory1[:min_length, 1], label='Original dy', color='C2')
#         ax2_left.set_ylabel('Original dy', color='C2')
#         ax2_left.tick_params(axis='y', labelcolor='C2')
        
#         # 右侧y轴显示稳定后轨迹
#         ax2_right = ax2.twinx()
#         line4 = ax2_right.plot(trajectory2[:min_length, 1], label='Stabilized dy', color='C3')
#         ax2_right.set_ylabel('Stabilized dy', color='C3')
#         ax2_right.tick_params(axis='y', labelcolor='C3')
        
#         # 合并图例
#         lines2 = line3 + line4
#         labels2 = [l.get_label() for l in lines2]
#         ax2_left.legend(lines2, labels2, loc='upper left')

#         # 设置x轴标签
#         ax2_left.set_xlabel('Frame Number')

#         fig.suptitle('Video Trajectory Comparison (Double Y-Axis)', x=0.15, y=0.96, ha='left')
#         fig.canvas.manager.set_window_title('Trajectory')

#         plt.tight_layout()
#         plt.savefig(output_path)
        
#         return fig, (ax1, ax2)


# 使用示例
if __name__ == "__main__":
    # fig, axes = plot_trajectory_comparison(
    #     video1_path='../data/1.avi',
    #     video2_path='../data/result.avi',
    #     scale_factor=1.0,
    #     output_path='./trajectory_plot.png'
    # )
    # plt.show()

    trajectory1, trajectory2, all_transforms1, all_transforms2 = gen_trajectory_data(video1_path='../data/1.avi', video2_path='../data/result.avi', resolution_option='native')
    trajectory_data = {
        'trajectory1': trajectory1,
        'trajectory2': trajectory2,
        'all_transforms1': all_transforms1,
        'all_transforms2': all_transforms2
    }
    figure = plt.figure(figsize=(8, 6)) # figure = Figure(figsize=(8, 6))
    figure = plot_trajectory_data(figure, trajectory_data)
    plt.show()
