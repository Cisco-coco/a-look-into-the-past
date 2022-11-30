import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from locate import locate
from sa_map import merge_SaliencyMap

def show(img, title="img"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def down(img, level=1):
    # Gaussian金字塔分解(Not DOG pyramid)
    for _ in range(level):
        img = cv2.pyrDown(img)
    return img

def up(img, level=1):
    for _ in range(level):
        img = cv2.pyrUp(img)
    return img

def merge_circle(src, dst, src_pts, H):
    # draw a circle and only save the circle part, then transform the circle part to the dst
    src_center = np.mean(src_pts, axis=0, dtype=np.int32)
    radius = (src.shape[0] + src.shape[1]) // 8
    area = np.zeros(src.shape, dtype=np.uint8)
    area = cv2.circle(area, tuple(src_center[0]), radius, 1, -1)
    src = area * src
    src_trans = cv2.warpPerspective(src, H, (src.shape[1], src.shape[0]))
    cv2.imwrite("imgs/result/trans.jpg", src)
    # show(src_trans, "src_trans")

    # merge
    result = np.copy(dst)
    result[src_trans > 0, :] = 0
    # show(result, "result")
    src_trans = cv2.cvtColor(src_trans, cv2.COLOR_GRAY2BGR)
    # show(src_trans, "bgr src_trans")
    result += src_trans    

    cv2.imwrite('imgs/result/result_raw.jpg', result)
    show(result, "result_raw")

    # blur boundary
    dst_center = np.dot(H, np.array([src_center[0][0], src_center[0][1], 1]))
    dst_center = dst_center / dst_center[2]
    dst_center = dst_center.astype(np.int32)[:2]
    print(f"dst_center: {dst_center}")
    area = np.ones_like(dst)
    area = cv2.circle(area, tuple(dst_center), int(radius*0.9), 0, -1)
    dst = area * dst
    blur = np.copy(result)
    blur[area==0] = 0
    crop = np.copy(result)
    crop[area==1] = 0         

    result = cv2.ximgproc.guidedFilter(dst, blur, 33, 2, -1) + crop
    show(result, "final result")

    return result

def merge_ellipse(src, dst, src_pts, H):
    # draw a circle and only save the circle part, then transform the circle part to the dst
    temp = cv2.warpPerspective(src, H, (src.shape[1], src.shape[0]))
    cv2.imwrite("imgs/result/warp.jpg", temp)
    src_center = np.mean(src_pts, axis=0, dtype=np.int32)
    radius = (src.shape[0] // 4 , src.shape[1] // 4) 
    area = np.zeros(src.shape, dtype=np.uint8)
    area = cv2.ellipse(area, tuple(src_center[0]), radius, angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
    src = area * src
    src_trans = cv2.warpPerspective(src, H, (src.shape[1], src.shape[0]))
    cv2.imwrite("imgs/result/src_masked.jpg", src)
    # show(src_trans, "src_trans")

    # merge
    result = np.copy(dst)
    result[src_trans > 0, :] = 0
    cv2.imwrite('imgs/result/dst_masked.jpg', result)
    # show(result, "result")
    src_trans = cv2.cvtColor(src_trans, cv2.COLOR_GRAY2BGR)
    # show(src_trans, "bgr src_trans")
    result += src_trans    

    cv2.imwrite('imgs/result/result_raw.jpg', result)
    show(result, "result_raw")

    # blur boundary
    dst_center = np.dot(H, np.array([src_center[0][0], src_center[0][1], 1]))
    dst_center = dst_center / dst_center[2]
    dst_center = dst_center.astype(np.int32)[:2]
    print(f"dst_center: {dst_center}")
    area = np.ones_like(dst)
    area = cv2.ellipse(area, tuple(dst_center), tuple(map(lambda x : int(0.9*x), radius)), angle=0, startAngle=0, endAngle=360, color=0, thickness=-1)
    # area = np.ones_like(src)
    # area = cv2.ellipse(area, tuple(src_center[0]), tuple(map(lambda x : int(0.9*x), radius)), angle=0, startAngle=0, endAngle=360, color=0, thickness=-1)
    # area = cv2.warpPerspective(area, H, (src.shape[1], src.shape[0]))
    # area = np.expand_dims(area,-1).repeat(3,axis=-1)
    
    dst = area * dst
    blur = np.copy(result)
    blur[area==0] = 0
    crop = np.copy(result)
    crop[area==1] = 0         
    
    cv2.imwrite('imgs/result/crop.jpg', crop)
    cv2.imwrite('imgs/result/before_blur.jpg', dst)
    result = cv2.ximgproc.guidedFilter(dst, blur, 33, 2, -1)
    cv2.imwrite('imgs/result/after_blur.jpg', result)
    result += crop
    # result = cv2.ximgproc.guidedFilter(dst, blur, 33, 2, -1) + crop
    show(result, "final result")

    return result

def homoConvert(past, present):
    surf = cv2.SIFT_create()
    # find the keypoints and descriptors with SURF
    kp1, des1 = surf.detectAndCompute(past, None)
    kp2, des2 = surf.detectAndCompute(present, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test, less is better
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 通过特征点坐标计算单应性矩阵H
    # (findHomography中使用了RANSAC算法剔初错误匹配)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    print(f"homo: {H}")
    # 使用单应性矩阵计算变换结果并绘图
    h, w = past.shape
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    # img2 = cv2.polylines(present, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # show(img2, "past after transform")
    # cv2.imwrite("imgs/result/transformed.jpg", img2)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv2.drawMatches(past, kp1, present, kp2, good, None, **draw_params)
    show(img3, "matches")
    cv2.imwrite('imgs/result/matches.jpg', img3)

    result = merge_ellipse(past, present, src_pts, H)
    return result

def blur_box(src_name, dst_name, box):
    src = cv2.imread(src_name)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.imread(dst_name)
    mask = np.zeros_like(src)
    mask[box[0]:box[2], box[1]:box[3]] = 1
    anti_mask = 1 - mask

    result_raw = src[:,:,np.newaxis] * mask[:,:,np.newaxis] + dst * anti_mask[:,:,np.newaxis]

    mid_h = (box[0] + box[2]) // 2
    mid_w = (box[1] + box[3]) // 2
    mask_h = int((box[2] - box[0]) * 0.45)
    mask_w = int((box[3] - box[1]) * 0.45)
    small_mask = np.zeros_like(src)
    small_mask[mid_h-mask_h:mid_h+mask_h, mid_w-mask_w:mid_w+mask_w] = 1
    small_anti_mask = 1 - small_mask
    blur = result_raw * small_anti_mask[:,:,np.newaxis]
    result = cv2.bilateralFilter(blur, 15, 75, 75)
    # result = cv2.medianBlur(blur, 3)
    result[small_mask==1] = 0
    crop = src[:,:,np.newaxis] * small_mask[:,:,np.newaxis]

    # plt.subplot(2,2,1)
    # plt.imshow(result_raw)
    # plt.subplot(2,2,2)
    # plt.imshow(blur)
    # plt.subplot(2,2,3)
    # plt.imshow(result)
    # plt.subplot(2,2,4)
    # plt.imshow(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))
    # plt.show()
    
    result += crop
    # show(result, "final result")

    return result

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.getcwd())))
    print(f"Current working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=float, default=1)
    parser.add_argument("--homoConvert", type=bool, default=False)
    args = parser.parse_args()
    past = cv2.imread("imgs/lbook.jpg")
    # show(past, "original past")
    past = cv2.cvtColor(past, cv2.COLOR_BGR2GRAY)
    present = cv2.imread("imgs/rbook.jpg")

    if args.homoConvert:
        result = homoConvert(past, present)
        cv2.imwrite('imgs/result/result.jpg', result)
    else:
        past_names = [f"imgs/gray/{x}.jpg" for x in range(1,50+1)]
        present_names = [f"imgs/rgb/{x}.jpg" for x in range(1,50+1)]
        boxes_list = locate(past_names)
        for index, (box, past, present) in enumerate(zip(boxes_list, past_names, present_names)):
            if box is None:
                result = merge_SaliencyMap([past], [present])[0]
                cv2.imwrite(f"imgs/result/{index+1}.jpg", result)
            else:
                result = blur_box(past, present, box)
                cv2.imwrite(f"imgs/result/{index+1}.jpg", result)
                
