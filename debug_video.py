import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# 添加全局缓存变量
plot_cache = {}

def create_plot_frame(x_indices, sliding_fft_t, sliding_fft_r, current_frame_idx, window_length, plot_width=800, plot_height=300):
    """
    创建当前帧对应的折线图（分为两个子图）
    
    参数:
    x_indices: 樣本點的橫座標數據（幀索引）
    sliding_fft_t: 平移穩定性分數序列
    sliding_fft_r: 旋轉穩定性分數序列
    current_frame_idx: 當前視頻幀索引
    window_length: 窗口長度
    plot_width: 圖像寬度
    plot_height: 圖像高度
    
    返回:
    numpy數組格式的圖像
    """
    # 检查缓存中是否已有该帧的图表
    cache_key = (current_frame_idx, plot_width, plot_height)
    if cache_key in plot_cache:
        return plot_cache[cache_key]
    
    # 創建圖形和兩個子圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(plot_width/100, plot_height/100), dpi=100)
    fig.suptitle(f'Stability Score (Window: {window_length})', fontsize=12)
    
    # 上子圖：平移穩定性分數
    ax1.plot(x_indices, sliding_fft_t, 'b-', linewidth=1, alpha=0.5, label='Translation')
    try:
        current_pos = np.where(x_indices <= current_frame_idx)[0][-1]
    except IndexError:
        current_pos = 0
    
    if current_pos > 0:
        ax1.plot(x_indices[:current_pos+1], sliding_fft_t[:current_pos+1], 'b-', linewidth=2.5)
    
    ax1.axvline(x=current_frame_idx, color='green', linestyle='--', linewidth=2, alpha=0.9)
    if current_pos < len(x_indices):
        ax1.plot(x_indices[current_pos], sliding_fft_t[current_pos], 'bo', markersize=6)
    
    ax1.set_ylabel('Translation', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(min(x_indices), max(x_indices))
    ax1.tick_params(axis='both', which='major', labelsize=8)
    
    # 下子圖：旋轉穩定性分數
    ax2.plot(x_indices, sliding_fft_r, 'r-', linewidth=1, alpha=0.5, label='Rotation')
    if current_pos > 0:
        ax2.plot(x_indices[:current_pos+1], sliding_fft_r[:current_pos+1], 'r-', linewidth=2.5)
    
    ax2.axvline(x=current_frame_idx, color='green', linestyle='--', linewidth=2, alpha=0.9)
    if current_pos < len(x_indices):
        ax2.plot(x_indices[current_pos], sliding_fft_r[current_pos], 'ro', markersize=6)
    
    ax2.set_ylabel('Rotation', fontsize=10)
    ax2.set_xlabel('Frame Index', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(min(x_indices), max(x_indices))
    ax2.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout(pad=0.5)
    
    # 將圖像轉換為numpy數組
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    
    # 使用opencv讀取圖像
    plot_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    
    plt.close(fig)
    buf.close()
    
    # 缓存最近几帧的图表（避免内存占用过大）
    if len(plot_cache) > 10:  # 只缓存最近10帧
        # 删除最早的缓存项
        oldest_key = next(iter(plot_cache))
        del plot_cache[oldest_key]
    
    plot_cache[cache_key] = plot_img
    return plot_img

def overlay_plot_on_video(video_path, data_path, output_path, window_length):
    """
    將穩定性分數折線圖實時疊加到視頻上並導出新視頻
    
    參數:
    video_path: 輸入視頻路徑
    data_path: numpy壓縮文件路徑（包含穩定性分數數據）
    output_path: 輸出視頻路徑
    window_length: 窗口長度
    """
    # 加載穩定性分數數據
    data = np.load(data_path)
    x_indices = data['x_indices']
    sliding_fft_t = data['sliding_fft_t']
    sliding_fft_r = data['sliding_fft_r']
    
    # 打開視頻文件
    cap = cv2.VideoCapture(video_path)
    
    # 獲取視頻屬性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 增大圖表區域佔比
    plot_width = int(width * 0.9)  # 占视频宽度的90%
    plot_height = int(height * 0.4)  # 占视频高度的40%（给两个图留更多空间）
    
    # 創建輸出視頻編寫器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print(f"Processing video with {total_frames} frames...")
    print(f"Chart size: {plot_width}x{plot_height}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 為當前幀創建穩定性分數圖表
        try:
            plot_img = create_plot_frame(
                x_indices, sliding_fft_t, sliding_fft_r, 
                frame_count, window_length, plot_width, plot_height
            )
            
            # 調整圖表大小（如果需要）
            if plot_img.shape[1] != plot_width or plot_img.shape[0] != plot_height:
                plot_img_resized = cv2.resize(plot_img, (plot_width, plot_height))
            else:
                plot_img_resized = plot_img
            
            # 將圖表疊加到視頻幀底部
            y_offset = height - plot_height
            x_offset = (width - plot_width) // 2  # 居中放置
            
            # 創建感興趣區域ROI
            roi = frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width]
            
            # 將圖表混合到視頻幀上
            # 创建alpha遮罩来处理可能的尺寸不匹配
            if roi.shape[:2] == plot_img_resized.shape[:2]:
                # 直接复制（提高性能）
                frame[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width] = plot_img_resized
            else:
                # 如果尺寸不匹配，则调整大小后复制
                plot_resized = cv2.resize(plot_img_resized, (roi.shape[1], roi.shape[0]))
                frame[y_offset:y_offset+plot_resized.shape[0], x_offset:x_offset+plot_resized.shape[1]] = plot_resized
            
        except Exception as e:
            print(f"Error creating plot for frame {frame_count}: {e}")
            pass
        
        # 寫入輸出視頻
        out.write(frame)
        
        frame_count += 1
        
        # 顯示進度
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # 釋放資源
    cap.release()
    out.release()
    
    # 清空缓存
    global plot_cache
    plot_cache.clear()
    
    print(f"Video processing completed. Output saved to: {output_path}")

def load_and_plot_sliding_fft_data(data_path, window_length):
    """
    从numpy压缩文件加载数据并绘制
    
    参数:
    data_path: numpy压缩文件路径
    window_length: 窗口长度
    """
    try:
        # 从单个.npz文件加载所有数据
        data = np.load(data_path)
        x_indices = data['x_indices']
        sliding_fft_t = data['sliding_fft_t']
        sliding_fft_r = data['sliding_fft_r']

        original_length = len(x_indices)
        
        # 绘制图表
        plt.ioff()  # 关闭交互模式
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制平移分量的滑动窗口FFT结果
        ax1.plot(x_indices, sliding_fft_t, 'b-', linewidth=1.5, marker='o', markersize=3)
        ax1.set_title(f'Translation Stability Score (Window Length: {window_length})')
        ax1.set_ylabel('Stability Score')
        ax1.grid(True, alpha=0.3)
        # 设置x轴范围与原始数据长度一致
        ax1.set_xlim(min(x_indices), max(x_indices))
        
        # 绘制旋转分量的滑动窗口FFT结果
        ax2.plot(x_indices, sliding_fft_r, 'r-', linewidth=1.5, marker='o', markersize=3)
        ax2.set_title(f'Rotation Stability Score (Window Length: {window_length})')
        ax2.set_xlabel('Frame Index (Window Center)')
        ax2.set_ylabel('Stability Score')
        ax2.grid(True, alpha=0.3)
        # 设置x轴范围与原始数据长度一致
        ax2.set_xlim(min(x_indices), max(x_indices))
        
        # 设置y轴范围固定
        ax1.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        # 保存图片
        # plt.savefig(f'logs/reloaded_sliding_window_fft_{window_length}.png')
        plt.close()
        
        print(f"Plot regenerated from saved data with window length {window_length}")
        
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
    except Exception as e:
        print(f"Error during loading or plotting: {e}")

if __name__ == '__main__':
    window_length = 80
    video_file_path = r"D:\Users\74055\Desktop\compare-stability\20251110_153808_158\VID_20251110_153808_158.mp4"
    data_file_path = f'logs/sliding_fft_data_window_{window_length}.npz'
    output_video_path = f'logs/stability_overlay_video_window_{window_length}.mp4'
    
    # 测试加载绘制折线图
    # load_and_plot_sliding_fft_data(f'logs/sliding_fft_data_window_{window_length}.npz', window_length)

    # 实时绘制折线图到视频
    overlay_plot_on_video(video_file_path, data_file_path, output_video_path, window_length)