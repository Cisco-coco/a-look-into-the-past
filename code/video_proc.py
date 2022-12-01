import cv2
import os
import numpy as np
from tqdm import tqdm

from main import homoConvert, show, merge_ellipse

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.getcwd())))
    print(f"Current working directory: {os.getcwd()}")

    pathway = cv2.imread("imgs/pathway_frame.jpg")
    pathway = cv2.cvtColor(pathway, cv2.COLOR_BGR2GRAY)
    show(pathway, "original pathway")

    videoReader = cv2.VideoCapture("video/pathway.mp4")
    total_frames = int(videoReader.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames)
    pbar.set_description("Process each frame as a rgb image")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter("video/result.avi", fourcc, 15, (pathway.shape[1], pathway.shape[0]))

    while True:
        ret, frame = videoReader.read()
        if not ret:
            break
        # cv2.imshow("frame", frame)
        result = homoConvert(pathway, frame)
        videoWriter.write(result)
        cv2.imshow("result", result)
        pbar.update(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoReader.release()
    videoWriter.release()