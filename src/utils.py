

def get_resize_dimensions(frame_width, frame_height, target_resolution) -> tuple[int, int]:
    """
    根据目标分辨率和视频方向计算调整后的尺寸
    """
    # 定义各种分辨率的宽度
    resolutions = {
        '4k': 3840,
        '1080p': 1920,
        '720p': 1280,
        '480p': 854
    }
    
    # 如果是原生分辨率，直接返回原始尺寸
    if target_resolution == 'native':
        return (int(frame_width), int(frame_height))
    
    # 获取目标宽度
    target_width = resolutions.get(target_resolution, frame_width)
    
    # 判断是横屏还是竖屏
    is_landscape = frame_width >= frame_height
    
    if is_landscape:
        # 横屏视频 - 以宽度为基准计算新尺寸
        new_width = target_width
        new_height = int(frame_height * (target_width / frame_width))
    else:
        # 竖屏视频 - 以高度为基准计算新尺寸
        new_height = target_width
        new_width = int(frame_width * (target_width / frame_height))
        
    return (int(new_width), int(new_height))