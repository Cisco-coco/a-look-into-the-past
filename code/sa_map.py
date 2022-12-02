import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def show(img, name="image"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(name)
    plt.show()

def findZone(img, threshold=0.2):
    saliencyAlgorithm = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliencyAlgorithm.computeSaliency(img)
    # print(f"salient map shape: {saliencyMap.shape}")
    maximum = np.max(saliencyMap)
    # print(f"maximum: {maximum}")
    # show(saliencyMap, "saliency map")
    biSaliencyMap = (saliencyMap > threshold * maximum).astype(np.uint8)
    # show(saliencyMap*255, "saliency map thresholded")
    
    # mask = np.zeros_like(saliencyMap, dtype=np.uint8)
    # h, w = saliencyMap.shape
    # start_h = h // 8
    # end_h = h * 7 // 8
    # start_w = w // 8
    # end_w = w * 7 // 8
    # mask[start_h:end_h, start_w:end_w] = 1
    # masked_biSaliencyMap = mask * biSaliencyMap

    # plt.subplot(2, 2, 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title("original")
    # plt.subplot(2, 2, 2)
    # plt.imshow(cv2.cvtColor(saliencyMap, cv2.COLOR_BGR2RGB))
    # plt.title("saliency map")
    # plt.subplot(2, 2, 3)
    # plt.imshow(cv2.cvtColor(biSaliencyMap*255, cv2.COLOR_BGR2RGB))
    # plt.title("saliency map thresholded")
    # plt.subplot(2, 2, 4)
    # plt.imshow(cv2.cvtColor(masked_biSaliencyMap*255, cv2.COLOR_BGR2RGB))
    # plt.title("masked saliency map")
    # plt.show()
    
    return biSaliencyMap


def merge_SaliencyMap_clone(src_names, dst_names, threshold=0.2):

    assert len(src_names) == len(dst_names), "src_names and dst_names should have the same length!"
    merged_list = []
    for src_name, dst_name in zip(src_names, dst_names):
        past = cv2.imread(src_name)
        # print(f"img.shape: {past.shape}")
        present = cv2.imread(dst_name)

        masked_biSaliencyMap = findZone(past)
        # print(f"dtype of masked_biSaliencyMap: {masked_biSaliencyMap.dtype}")

        mask  = masked_biSaliencyMap * 255

        result = cv2.seamlessClone(past, present, mask, (past.shape[1]//2, past.shape[0]//2), cv2.NORMAL_CLONE)
        merged_list.append(result)
        # show(merged, "merged")

    return merged_list

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.getcwd()))
    print(f"Current working directory: {os.getcwd()}")
    src_names = ["imgs/gray/1.jpg"]
    dst_names = ["imgs/rgb/1.jpg"]
    merged = merge_SaliencyMap_clone(src_names, dst_names)