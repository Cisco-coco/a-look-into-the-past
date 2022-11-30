import os
import numpy as np
from PIL import Image

import sys
sys.path.append("..")
from yolo import YOLO

def locate(img_list):
    yolo = YOLO()
    crop = False
    count = False
    boxes_list = []

    def convert(box):
        top, left, bottom, right = box

        top = np.array(top)
        left = np.array(left)
        bottom = np.array(bottom)
        right = np.array(right)

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))

        return top, left, bottom, right

    for img in img_list:
        try:
            image = Image.open(img)
        except:
            print(f'image {img} cannot be opened!')
            continue
        else:
            print(f"image {img} opened successfully!")
            r_image, boxes = yolo.detect_image(image, crop, count)
            # r_image.show()
            if boxes is None:
                boxes_list.append(None)
                continue
            pos_list = []
            for box in boxes:
                pos_list.append(convert(box))
            top = min(pos_list[i][0] for i in range(len(pos_list)))
            left = min(pos_list[i][1] for i in range(len(pos_list)))
            bottom = max(pos_list[i][2] for i in range(len(pos_list)))
            right = max(pos_list[i][3] for i in range(len(pos_list)))
            boxes_list.append([top, left, bottom, right])

    return boxes_list

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.getcwd())))
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    img_list = ['imgs/gray/1.jpg', 'imgs/gray/2.jpg', 'imgs/gray/3.jpg']
    boxes_list = locate(img_list)
    print(boxes_list)
