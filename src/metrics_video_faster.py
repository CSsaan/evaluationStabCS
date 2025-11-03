import os
import sys
import numpy as np
import cv2
import math
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt

def metrics(original_video, pred_video, scale_factor=1.0, visualize=False):
    # 打开视频文件
    cap1 = cv2.VideoCapture(original_video)
    cap2 = cv2.VideoCapture(pred_video)
    
    # 获取视频长度（帧数）
    length = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                 int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    # 使用SIFT特征检测
    try:
        sift = cv2.SIFT_create(nfeatures=1000) # 新版本OpenCV (4.5.0以后) - SIFT已移至主模块
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000) # 旧版本OpenCV - 使用contrib模块
        
    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    # 降低分辨率处理
    SCALE_FACTOR = scale_factor
    # 跳帧处理，每隔N帧处理一次
    FRAME_SKIP = 1  # 设置为1表示每帧都处理，设置为2表示每隔一帧处理
  
    # Apply the homography transformation if we have enough good matches 
    MIN_MATCH_COUNT = 10
  
    ratio = 0.7
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
        cap1.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret1, img1 = cap1.read()
        ret2, img1o = cap2.read()
        
        # 检查是否成功读取帧
        if not ret1 or not ret2:
            break
        
        # 保存原始尺寸用于可视化
        original_img1 = img1.copy()
        original_img1o = img1o.copy()
        
        # 下采样
        if SCALE_FACTOR != 1.0:
            img1 = cv2.resize(img1, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            img1o = cv2.resize(img1o, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

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
        w, _ = np.linalg.eig(M[0:2, 0:2])
        w = np.sort(w)[::-1]
        DV = w[1]/w[0]
	    
        CR_seq.append(1/scaleRecovered)
        DV_seq.append(DV)
        
        # 实时可视化显示
        if visualize:
            # 在原始尺寸图像上绘制关键点和匹配
            vis_img1 = original_img1.copy()
            vis_img1o = original_img1o.copy()
            
            # 调整关键点坐标到原始尺寸
            scale_x = original_img1.shape[1] / img1.shape[1]
            scale_y = original_img1.shape[0] / img1.shape[0]
            
            # 绘制关键点
            for kp in keyPoints1:
                pt = (int(kp.pt[0] * scale_x), int(kp.pt[1] * scale_y))
                cv2.circle(vis_img1, pt, 3, (0, 255, 0), -1)
                
            for kp in keyPoints1o:
                pt = (int(kp.pt[0] * scale_x), int(kp.pt[1] * scale_y))
                cv2.circle(vis_img1o, pt, 3, (0, 255, 0), -1)
            
            # 拼接两幅图像用于显示（使用填充而不是缩放来保持宽高比）
            height1, width1 = vis_img1.shape[:2]
            height2, width2 = vis_img1o.shape[:2]
            
            if height1 != height2:
                target_height = max(height1, height2)
                
                # 为较矮的图像添加填充
                if height1 < target_height:
                    padding = target_height - height1
                    vis_img1 = cv2.copyMakeBorder(vis_img1, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                if height2 < target_height:
                    padding = target_height - height2
                    vis_img1o = cv2.copyMakeBorder(vis_img1o, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            combined_img = np.hstack((vis_img1, vis_img1o))
            
            # 绘制匹配线
            for match in goodMatches[:20]:  # 只显示前20个匹配
                pt1 = keyPoints1[match.queryIdx].pt
                pt2 = keyPoints1o[match.trainIdx].pt
                
                # 调整坐标到拼接图像中的位置
                start_pt = (int(pt1[0] * scale_x), int(pt1[1] * scale_y))
                end_pt = (int(pt2[0] * scale_x + vis_img1.shape[1]), int(pt2[1] * scale_y))
                
                cv2.line(combined_img, start_pt, end_pt, (0, 255, 255), 1)

            # 添加文字信息
            cv2.putText(combined_img, f'Frame: {i}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(combined_img, f'Matches: {len(goodMatches)}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(combined_img, f'Scale: {1/scaleRecovered:.2f}', (10, 110), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示图像
            cv2.imshow('Feature Matches', combined_img)
            
            # 按'q'键退出可视化
            if cv2.waitKey(1) & 0xFF == ord('q'):
                visualize = False
                cv2.destroyWindow('Feature Matches')
        
        # For Stability score calculation
        if i+1 < length:
            # 读取下一帧
            cap2.set(cv2.CAP_PROP_POS_FRAMES, i+1)
            ret2, img2o = cap2.read()
            if not ret2:
                break

            # 下采样
            if SCALE_FACTOR != 1.0:
                img2o = cv2.resize(img2o, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            # 转换为灰度图
            gray2o = cv2.cvtColor(img2o, cv2.COLOR_BGR2GRAY)
            
            keyPoints2o, descriptors2o = sift.detectAndCompute(gray2o, None)
            matches = bf.knnMatch(descriptors1o, descriptors2o, k=2)
            goodMatches_stability = []
		    
            for m, n in matches:
                if m.distance < ratio * n.distance:
                  goodMatches_stability.append(m)
		        
            if len(goodMatches_stability) > MIN_MATCH_COUNT:
                # Get the good key points positions
                sourcePoints = np.float32([ keyPoints1o[m.queryIdx].pt for m in goodMatches_stability ]).reshape(-1, 1, 2)
                destinationPoints = np.float32([ keyPoints2o[m.trainIdx].pt for m in goodMatches_stability ]).reshape(-1, 1, 2)
                
                # Obtain the homography matrix
                M, _ = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
                # M, _ = cv2.estimateAffine2D(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=thresh)
            else:
                continue
		    
            P_seq.append(np.matmul(Pt, M))
            Pt = np.matmul(Pt, M)

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
        fft_t = np.fft.fft(P_seq_t)
        fft_r = np.fft.fft(P_seq_r)
        fft_t = np.abs(fft_t)**2  
        fft_r = np.abs(fft_r)**2

        # freq = np.fft.fftfreq(len(P_seq_t))
        # plt.plot(freq, abs(fft_r)**2)
        # plt.show()
        # print(abs(fft_r)**2)
        # print(freq)
        
        fft_t = np.delete(fft_t, 0)
        fft_r = np.delete(fft_r, 0)
        fft_t = fft_t[:len(fft_t)//2]
        fft_r = fft_r[:len(fft_r)//2]
      
        SS_t = np.sum(fft_t[:5])/np.sum(fft_t)
        SS_r = np.sum(fft_r[:5])/np.sum(fft_r)
       
        # Print results
        print('\n')
        print('*' * 60)
        print(' Cropping ratio ↑ Avg, (Min):')
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
        print(    str.format('{0:.4f}', np.min([np.mean(CR_seq), 1])) +' ('+ str.format('{0:.4f}', np.min([np.min(CR_seq), 1])) +') ' )

        print(' Distortion value ↑ :')
        '''
        含义：
            表示视频稳定过程中产生的图像畸变程度
            通过对单应性矩阵的特征值分析得到：DV = w[1]/w[0]（w是排序后的特征值）
        解读：
            值越接近1表示畸变越小
            值越小表示畸变越严重
            取最小值的绝对值作为最终指标
        '''
        print(    str.format('{0:.4f}', np.absolute(np.min(DV_seq))) )

        print(' StabilityScore ↑ Avg, (Trans, Rot):')
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
        print(    str.format('{0:.4f}',  (SS_t+SS_r)/2) +' (' + str.format('{0:.4f}', SS_t) +', '+ str.format('{0:.4f}', SS_r) +') ' )
        print('*' * 60)

        if len(CR_seq) > 0:
            CR_AVG = np.min([np.mean(CR_seq), 1])
            CR_MIN = np.min([np.min(CR_seq), 1])
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
            CR_AVG = np.min([np.mean(CR_seq), 1])
            CR_MIN = np.min([np.min(CR_seq), 1])
        else:
            CR_AVG = np.nan
            CR_MIN = np.nan
        if len(DV_seq) > 0:
            DVDV = np.absolute(np.min(DV_seq))
        else:
            DVDV = np.nan
        return (CR_AVG, CR_MIN), DVDV, (np.nan, np.nan, np.nan)


if __name__ == '__main__':
    CR_AVG_MIN, DVDV, SS_AVG_T_R = metrics(original_video='../data/1.avi', pred_video='../data/result.avi', scale_factor=0.5, visualize=True)
    print(f"CroppingRatio(average, min)↑: {CR_AVG_MIN[0]}, {CR_AVG_MIN[1]}")
    print(f"DirectionalVariation↑: {DVDV}")
    print(f"StabilityScore(average, trans, rotate)↑: {SS_AVG_T_R[0]}, {SS_AVG_T_R[1]}, {SS_AVG_T_R[2]}")
