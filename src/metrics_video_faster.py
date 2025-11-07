import os
import sys
import numpy as np
import cv2
import math
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import get_resize_dimensions


def metrics1(original_video, pred_video, resolution_option='native', visualize=True):
    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    sift = cv2.SIFT_create()

    # Apply the homography transformation if we have enough good matches
    MIN_MATCH_COUNT = 10

    ratio = 0.6
    thresh = 5.0

    CR_seq = []
    DV_seq = []
    Pt = np.eye(3)
    P_seq = []

    vc_o = cv2.VideoCapture(original_video)
    vc_p = cv2.VideoCapture(pred_video)

    # 创建可视化窗口
    if visualize:
        cv2.namedWindow('Feature Matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Feature Matches', 1200, 600)

    rval_o = vc_o.isOpened()
    rval_p = vc_p.isOpened()
    if not rval_o or not rval_p:
        raise Exception('Cannot open video file')
    
    # 根据预设分辨率下采样
    (target_width, target_height) = get_resize_dimensions(vc_o.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                       vc_o.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                                       resolution_option)
    print(f'Use Resolution: {target_width}x{target_height}')

    imgs1 = []
    imgs1o = []
    while (rval_o and rval_p):
        rval_o, img1 = vc_o.read()
        rval_p, img1o = vc_p.read()
        
        if rval_o and rval_p:
            if resolution_option != 'native': # 调整图像大小
                img1 = cv2.resize(img1, (target_width, target_height))
            img1o = cv2.resize(img1o, (target_width, target_height))
            imgs1.append(img1)
            imgs1o.append(img1o)

    is_got_bad_item = False
    for i in tqdm(range(len(imgs1)), desc="Processing frames"):
        # Load the images in gray scale
        img1 = imgs1[i]
        img1o = imgs1o[i]

        # Detect the SIFT key points and compute the descriptors for the two images
        keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keyPoints1o, descriptors1o = sift.detectAndCompute(img1o, None)

        # Match the descriptors
        matches = bf.knnMatch(descriptors1, descriptors1o, k=2)

        # Select the good matches using the ratio test
        goodMatches = []

        for m, n in matches:
            if m.distance < ratio * n.distance:
                goodMatches.append(m)

        if len(goodMatches) > MIN_MATCH_COUNT:
            # Get the good key points positions
            sourcePoints = np.float32([
                keyPoints1[m.queryIdx].pt for m in goodMatches
            ]).reshape(-1, 1, 2)
            destinationPoints = np.float32([
                keyPoints1o[m.trainIdx].pt for m in goodMatches
            ]).reshape(-1, 1, 2)

            # Obtain the homography matrix
            M, _ = cv2.findHomography(
                sourcePoints,
                destinationPoints,
                method=cv2.RANSAC,
                ransacReprojThreshold=thresh)
        else:
            is_got_bad_item = True

        if not is_got_bad_item:
            # Obtain Scale, Translation, Rotation, Distortion value
            # Based on https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix
            scaleRecovered = np.sqrt(M[0, 1]**2 + M[0, 0]**2)

            w, _ = np.linalg.eig(M[0:2, 0:2])
            w = np.sort(w)[::-1]
            DV = w[1] / w[0]

            CR_seq.append(1 / scaleRecovered)
            DV_seq.append(DV)

            # 绘制匹配结果
            img_matches = cv2.drawMatchesKnn(img1, keyPoints1, img1o, keyPoints1o, [goodMatches[:20]], None, flags=2)
            cv2.imshow("Feature Matches", img_matches)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow('Feature Matches')
                break
            if cv2.getWindowProperty('Feature Matches', cv2.WND_PROP_VISIBLE) < 1:
                break

            # For Stability score calculation
            if i + 1 < len(imgs1):
                img2o = imgs1o[i + 1]

                keyPoints2o, descriptors2o = sift.detectAndCompute(img2o, None)
                matches = bf.knnMatch(descriptors1o, descriptors2o, k=2)
                goodMatches = []

                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        goodMatches.append(m)

                if len(goodMatches) > MIN_MATCH_COUNT:
                    # Get the good key points positions
                    sourcePoints = np.float32([
                        keyPoints1o[m.queryIdx].pt for m in goodMatches
                    ]).reshape(-1, 1, 2)
                    destinationPoints = np.float32([
                        keyPoints2o[m.trainIdx].pt for m in goodMatches
                    ]).reshape(-1, 1, 2)

                    # Obtain the homography matrix
                    M, _ = cv2.findHomography(
                        sourcePoints,
                        destinationPoints,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=thresh)

                P_seq.append(np.matmul(Pt, M))
                Pt = np.matmul(Pt, M)

    if is_got_bad_item:
        return (np.nan, np.nan), np.nan, (np.nan, np.nan, np.nan)

    # 释放视频捕获对象
    vc_o.release()
    vc_p.release()
    if visualize:
        cv2.destroyAllWindows()

    # Make 1D temporal signals
    P_seq_t = []
    P_seq_r = []

    for Mp in P_seq:
        transRecovered = np.sqrt(Mp[0, 2]**2 + Mp[1, 2]**2)
        # Based on https://math.stackexchange.com/questions/78137/decomposition-of-a-nonsquare-affine-matrix
        thetaRecovered = np.arctan2(Mp[1, 0], Mp[0, 0]) * 180 / np.pi
        P_seq_t.append(transRecovered)
        P_seq_r.append(thetaRecovered)

    # FFT
    fft_t = np.fft.fft(P_seq_t)
    fft_r = np.fft.fft(P_seq_r)
    fft_t = np.abs(fft_t)**2
    fft_r = np.abs(fft_r)**2

    fft_t = np.delete(fft_t, 0)
    fft_r = np.delete(fft_r, 0)
    fft_t = fft_t[:len(fft_t) // 2]
    fft_r = fft_r[:len(fft_r) // 2]

    SS_t = np.sum(fft_t[:5]) / np.sum(fft_t)
    SS_r = np.sum(fft_r[:5]) / np.sum(fft_r)

    # Print results
    print('\n')
    print('*' * 60)
    print('Cropping ratio ↑ Avg, (Min):')
    '''
    含义：
        代表视频稳定过程中视场的缩小程度
        通过计算单应性矩阵的尺度分量得到：scaleRecovered = np.sqrt(M[0,1]**2 + M[0,0]**2)
        CR_seq.append(1/scaleRecovered) 表示相对于原始尺度的逆比例
    解读：
        值越小表示裁剪越严重（画面缩小越多）
        平均值反映整体裁剪程度
        最小值反映最严重裁剪的帧
    '''
    print(str.format('{0:.4f}', np.min([np.mean(CR_seq), 1])) +' ('+ str.format('{0:.4f}', np.min([np.min(CR_seq), 1])) +') ' )

    print('Distortion value ↑ :')
    '''
    含义：
        表示视频稳定过程中产生的图像畸变程度
        通过对单应性矩阵的特征值分析得到：DV = w[1]/w[0]（w是排序后的特征值）
    解读：
        值越接近1表示畸变越小
        值越小表示畸变越严重
        取最小值的绝对值作为最终指标
    '''
    print(str.format('{0:.4f}', np.absolute(np.min(DV_seq))) )

    print('StabilityScore ↑ Avg, (Trans, Rot):')
    '''
    含义：
        综合评估视频稳定效果的指标
        通过分析帧间变换的频域特性计算得出
    具体计算：
        计算相邻帧之间的平移(translation)和旋转(rotation)变化
        对这些变化进行傅里叶变换得到频域表示
        计算低频成分占比：
        SS_t = np.sum(fft_t[:5])/np.sum(fft_t) (平移稳定性)
        SS_r = np.sum(fft_r[:5])/np.sum(fft_r) (旋转稳定性)
        平均值 (SS_t+SS_r)/2 为综合稳定性评分
    解读：
        值越高表示稳定性越好
        平移和旋转分量分别反映在对应方向上的稳定性
        低频成分占比高表示变化平缓，稳定性好
    '''
    print(str.format('{0:.4f}',  (SS_t+SS_r)/2) +' (' + str.format('{0:.4f}', SS_t) +', '+ str.format('{0:.4f}', SS_r) +') ' )
    print('*' * 60)

    print(np.min([np.mean(CR_seq), 1]), np.absolute(np.min(DV_seq)), (SS_t + SS_r) / 2) 

    CR_AVG = np.min([np.mean(CR_seq), 1])
    CR_MIN = np.min([np.min(CR_seq), 1])
    DVDV = np.absolute(np.min(DV_seq))
    SS_avg, SS_t, SS_r = (SS_t+SS_r)/2, SS_t, SS_r

    return (CR_AVG, CR_MIN), DVDV, (SS_avg, SS_t, SS_r)


def metrics(original_video, pred_video, resolution_option='native', visualize=False):
    # 打开视频文件
    cap1 = cv2.VideoCapture(original_video)
    cap2 = cv2.VideoCapture(pred_video)
    
    # 获取视频长度（帧数）
    length = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                 int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    # 使用SIFT特征检测
    try:
        sift = cv2.SIFT_create(nfeatures=500) # 新版本OpenCV (4.5.0以后) - SIFT已移至主模块 nfeatures=1000
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=500) # 旧版本OpenCV - 使用contrib模块 nfeatures=1000
        
    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    # 降低分辨率处理
    TARGET_RESOLUTION = resolution_option
    # 跳帧处理，每隔N帧处理一次
    FRAME_SKIP = 1  # 设置为1表示每帧都处理，设置为2表示每隔一帧处理
  
    # Apply the homography transformation if we have enough good matches 
    MIN_MATCH_COUNT = 10
  
    ratio = 0.6
    thresh = 5.0
  
    CR_seq = []
    DV_seq = []
    Pt = np.eye(3)
    P_seq = []
    
    # 创建可视化窗口
    if visualize:
        cv2.namedWindow('Feature Matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Feature Matches', 1200, 600)
  
    for i in tqdm(range(0, length, FRAME_SKIP), desc="Processing frames"):
        # 读取视频帧
        ret1, img1 = cap1.read()
        ret2, img1o = cap2.read()
        if not ret1 or not ret2:
            print(f"Error reading frames:{i} or end of video reached.")
            continue
        
        # 根据预设分辨率下采样
        # if TARGET_RESOLUTION != 'native':
        # 获取目标尺寸
        (target_width, target_height) = get_resize_dimensions(img1.shape[1], img1.shape[0], TARGET_RESOLUTION)
            
        # 调整图像大小
        if TARGET_RESOLUTION != 'native':
            img1 = cv2.resize(img1, (target_width, target_height))
        img1o = cv2.resize(img1o, (target_width, target_height))

        # 转换为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray1o = cv2.cvtColor(img1o, cv2.COLOR_BGR2GRAY)

        # Detect the SIFT key points and compute the descriptors for the two images
        keyPoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keyPoints1o, descriptors1o = sift.detectAndCompute(gray1o, None)
        
        # Match the descriptors
        if descriptors1 is not None and descriptors1o is not None:
            matches = bf.knnMatch(descriptors1, descriptors1o, k=2)
        else:
            matches = None
            
        if matches is None:
            continue
  
        # Select the good matches using the ratio test
        goodMatches = []
        """
        for m, n in matches:
            if m.distance < ratio * n.distance:
                goodMatches.append(m)
        """
        for m_n in matches:
            if len(m_n) > 1:
                m = m_n[0]
                n = m_n[1]
                if m.distance < ratio * n.distance:
                    goodMatches.append(m)
        
        if len(goodMatches) > MIN_MATCH_COUNT:
            # Get the good key points positions
            sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
            destinationPoints = np.float32([ keyPoints1o[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
            
            # Obtain the homography matrix
            M, _ = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
        else:
            continue
            
        if M is None:
            continue
  
        # Obtain Scale, Translation, Rotation, Distortion value
        scaleRecovered = np.sqrt(M[0,1]**2 + M[0,0]**2)
        w, _ = np.linalg.eig(M[0:2, 0:2]) # 这里对单应性矩阵M 的前2×2子矩阵进行特征值分
        w = np.sort(w)[::-1] # 两个特征值 w[0] 和 w[1]（已排序）分别代表变换在两个正交主方向上的缩放程度
        DV = w[1]/w[0] # 它们的比值反映了两个主方向之间缩放的不平衡程度
	    
        CR_seq.append(1/scaleRecovered)
        DV_seq.append(DV)
        
        # 绘制匹配结果
        img_matches = cv2.drawMatchesKnn(img1, keyPoints1, img1o, keyPoints1o, [goodMatches[:20]], None, flags=2)
        cv2.imshow("Feature Matches", img_matches)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow('Feature Matches')
            break
        if cv2.getWindowProperty('Feature Matches', cv2.WND_PROP_VISIBLE) < 1:
            break
        

        # For Stability score calculation
        if i > 0:
            matches = bf.knnMatch(before_descriptors1o, descriptors1o, k=2)
            goodMatches_stability = []
		    
            for m, n in matches:
                if m.distance < ratio * n.distance:
                  goodMatches_stability.append(m)
		        
            if len(goodMatches_stability) > MIN_MATCH_COUNT:
                # Get the good key points positions
                sourcePoints = np.float32([ before_keyPoints1o[m.queryIdx].pt for m in goodMatches_stability ]).reshape(-1, 1, 2)
                destinationPoints = np.float32([ keyPoints1o[m.trainIdx].pt for m in goodMatches_stability ]).reshape(-1, 1, 2)
                
                # Obtain the homography matrix
                M, _ = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
                # M, _ = cv2.estimateAffine2D(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
            else:
                continue
		    
            P_seq.append(np.matmul(Pt, M))
            Pt = np.matmul(Pt, M)
        before_keyPoints1o = keyPoints1o
        before_descriptors1o = descriptors1o.copy() if descriptors1o is not None else None

    # 释放视频捕获对象
    cap1.release()
    cap2.release()
    
    # 关闭可视化窗口
    if visualize:
        cv2.destroyAllWindows()
    
    # Make 1D temporal signals
    P_seq_t = []
    P_seq_r = []
    
    for Mp in P_seq:
        transRecovered = np.sqrt(Mp[0, 2]**2 + Mp[1, 2]**2)
        thetaRecovered = np.arctan2(Mp[1, 0], Mp[0, 0]) * 180 / np.pi
        P_seq_t.append(transRecovered)
        P_seq_r.append(thetaRecovered)
    
    if len(P_seq) > 0:
        # FFT
        fft_t = np.abs(np.fft.fft(P_seq_t))**2
        fft_r = np.abs(np.fft.fft(P_seq_r))**2

        fft_t = np.delete(fft_t, 0)[:len(fft_t)//2]
        fft_r = np.delete(fft_r, 0)[:len(fft_r)//2]

        SS_t = np.sum(fft_t[:5])/np.sum(fft_t)
        SS_r = np.sum(fft_r[:5])/np.sum(fft_r)
       
        # Print results
        print('\n')
        print('*' * 60)
        print('Cropping ratio ↑ Avg, (Min):')
        '''
        含义：
            代表视频稳定过程中视场的缩小程度
            通过计算单应性矩阵的尺度分量得到：scaleRecovered = np.sqrt(M[0,1]**2 + M[0,0]**2)
            CR_seq.append(1/scaleRecovered) 表示相对于原始尺度的逆比例
        解读：
            值越小表示裁剪越严重（画面缩小越多）
            平均值反映整体裁剪程度
            最小值反映最严重裁剪的帧
        '''
        CR_seq_mean_ori = np.min([np.mean(CR_seq), 1])
        CR_seq_min_ori = np.min([np.min(CR_seq), 1])
        # 取倒数
        CR_seq_reciprocals = [1/x for x in CR_seq]
        CR_seq_mean_reciprocals = np.min([np.mean(CR_seq_reciprocals), 1])
        CR_seq_min_reciprocals = np.min([np.min(CR_seq_reciprocals), 1])
        # 取最小值
        if CR_seq_mean_ori <= CR_seq_mean_reciprocals:
            CR_seq_mean = CR_seq_mean_ori
            CR_seq_min = CR_seq_min_ori
        else:
            CR_seq_mean = CR_seq_mean_reciprocals
            CR_seq_min = CR_seq_min_reciprocals
        print('  ' + str.format('{0:.4f}', CR_seq_mean) +' ('+ str.format('{0:.4f}', CR_seq_min) +') ' )
        
        print('Distortion value ↑ :')
        '''
        含义：
            表示视频稳定过程中产生的图像畸变程度
            通过对单应性矩阵的特征值分析得到：DV = w[1]/w[0]（w是排序后的特征值）
        解读：
            值越接近1表示畸变越小
            值越小表示畸变越严重
            取最小值的绝对值作为最终指标
        '''
        print('  ' + str.format('{0:.4f}', np.absolute(np.min(DV_seq))) )

        print('StabilityScore ↑ Avg, (Trans, Rot):')
        '''
        含义：
            综合评估视频稳定效果的指标
            通过分析帧间变换的频域特性计算得出
        具体计算：
            计算相邻帧之间的平移(translation)和旋转(rotation)变化
            对这些变化进行傅里叶变换得到频域表示
            计算低频成分占比：
            SS_t = np.sum(fft_t[:5])/np.sum(fft_t) (平移稳定性)
            SS_r = np.sum(fft_r[:5])/np.sum(fft_r) (旋转稳定性)
            平均值 (SS_t+SS_r)/2 为综合稳定性评分
        解读：
            值越高表示稳定性越好
            平移和旋转分量分别反映在对应方向上的稳定性
            低频成分占比高表示变化平缓，稳定性好
        '''
        print('  ' + str.format('{0:.4f}',  (SS_t+SS_r)/2) +' (' + str.format('{0:.4f}', SS_t) +', '+ str.format('{0:.4f}', SS_r) +') ' )
        print('*' * 60)

        if len(CR_seq) > 0:
            CR_AVG = CR_seq_mean
            CR_MIN = CR_seq_min
        else:
            CR_AVG = np.nan
            CR_MIN = np.nan
        if len(DV_seq) > 0:
            DVDV = np.absolute(np.min(DV_seq))
        else:
            DVDV = np.nan
        
        SS_avg, SS_t, SS_r = (SS_t+SS_r)/2, SS_t, SS_r

        return (CR_AVG, CR_MIN), DVDV, (SS_avg, SS_t, SS_r)

    else:
        if len(CR_seq) > 0:
            CR_AVG = CR_seq_mean
            CR_MIN = CR_seq_min
        else:
            CR_AVG = np.nan
            CR_MIN = np.nan
        if len(DV_seq) > 0:
            DVDV = np.absolute(np.min(DV_seq))
        else:
            DVDV = np.nan
        return (CR_AVG, CR_MIN), DVDV, (np.nan, np.nan, np.nan)


if __name__ == '__main__':
    CR_AVG_MIN, DVDV, SS_AVG_T_R = metrics(original_video='../../data/1.avi', pred_video='../../data/result.avi', resolution_option='native', visualize=True)
    print(f"CroppingRatio(average, min)↑: {CR_AVG_MIN[0]}, {CR_AVG_MIN[1]}")
    print(f"DirectionalVariation↑: {DVDV}")
    print(f"StabilityScore(average, trans, rotate)↑: {SS_AVG_T_R[0]}, {SS_AVG_T_R[1]}, {SS_AVG_T_R[2]}")


'''
************************************************************
Cropping ratio ↑ Avg, (Min):
1.0000 (1.0000)
Distortion value ↑ :
0.8746
StabilityScore ↑ Avg, (Trans, Rot):
0.6933 (0.6540, 0.7325/0.8744)
************************************************************
(np.float64(1.0), np.float64(0.8731453968997918), np.float64(0.7008247105517187))
'''