import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from locate import locate

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
    src_center = np.mean(src_pts, axis=0, dtype=np.int32)
    radius = (src.shape[0] // 4 , src.shape[1] // 4) 
    area = np.zeros(src.shape, dtype=np.uint8)
    area = cv2.ellipse(area, tuple(src_center[0]), radius, angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
    src = area * src
    src_trans = cv2.warpPerspective(src, H, (src.shape[1], src.shape[0]))
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
    area = cv2.ellipse(area, tuple(dst_center), tuple(map(lambda x : int(0.9*x), radius)), angle=0, startAngle=0, endAngle=360, color=0, thickness=-1)
    dst = area * dst
    blur = np.copy(result)
    blur[area==0] = 0
    crop = np.copy(result)
    crop[area==1] = 0         

    result = cv2.ximgproc.guidedFilter(dst, blur, 33, 2, -1) + crop
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
    # show(img2)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    # img3 = cv2.drawMatches(past, kp1, present, kp2, good, None, **draw_params)
    # show(img3, "matches")
    # cv2.imwrite('imgs/img3.jpg', img3)

    result = merge_ellipse(past, present, src_pts, H)
    return result

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.getcwd())))
    print(f"Current working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=float, default=1)
    parser.add_argument("--homoConvert", type=bool, default=False)
    args = parser.parse_args()
    past = cv2.imread("imgs/gray/3.jpg")
    past = cv2.cvtColor(past, cv2.COLOR_BGR2GRAY)
    show(past, "original past")
    present = cv2.imread("imgs/rgb/3.jpg")

    if args.homoConvert:
        result = homoConvert(past, present)
        cv2.imwrite('imgs/result/result.jpg', result)
    else:
        box = locate(['imgs/gray/3.jpg'])[0]
        mask = np.zeros_like(past)
        mask[box[0]:box[2], box[1]:box[3]] = 1
        anti_mask = 1 - mask
        result = present * anti_mask[:,:,np.newaxis] + past[:,:,np.newaxis] * mask[:,:,np.newaxis]
        show(result, "result")
        cv2.imwrite("imgs/result/3.jpg", result)
