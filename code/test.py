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
    print(f"salient map shape: {saliencyMap.shape}")
    maximum = np.max(saliencyMap)
    print(f"maximum: {maximum}")
    # show(saliencyMap, "saliency map")
    biSaliencyMap = (saliencyMap > threshold * maximum).astype(np.uint8)
    # show(saliencyMap*255, "saliency map thresholded")
    mask = np.zeros_like(saliencyMap, dtype=np.uint8)
    h, w = saliencyMap.shape
    start_h = h // 8
    end_h = h * 7 // 8
    start_w = w // 8
    end_w = w * 7 // 8
    mask[start_h:end_h, start_w:end_w] = 1
    masked_biSaliencyMap = mask * biSaliencyMap

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("original")
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(saliencyMap, cv2.COLOR_BGR2RGB))
    plt.title("saliency map")
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(biSaliencyMap*255, cv2.COLOR_BGR2RGB))
    plt.title("saliency map thresholded")
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(masked_biSaliencyMap*255, cv2.COLOR_BGR2RGB))
    plt.title("masked saliency map")
    plt.show()
    
    return biSaliencyMap


os.chdir(os.path.join(os.path.dirname(os.getcwd())))
print(f"Current working directory: {os.getcwd()}")

past = cv2.imread("imgs/gray/1.jpg")
print(f"img.shape: {past.shape}")
present = cv2.imread("imgs/rgb/1.jpg")

masked_biSaliencyMap = findZone(past)
print(f"dtype of masked_biSaliencyMap: {masked_biSaliencyMap.dtype}")

masked_biSaliencyMap = masked_biSaliencyMap.astype(np.uint8)
print(f"maximum of masked_biSaliencyMap: {np.max(masked_biSaliencyMap)}")
anti_mask = 1 - masked_biSaliencyMap
print(f"maximum of masked_present: {np.max(anti_mask)}")
masked_present = present * anti_mask[:, :, np.newaxis]
crop = past * masked_biSaliencyMap[:, :, np.newaxis]

merged = crop + masked_present
show(merged, "merged") 