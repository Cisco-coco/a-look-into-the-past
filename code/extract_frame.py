import cv2
import os
from tqdm import tqdm
import shutil

def extract(path, save_all=False):
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = [0, 0]
    rgb_count = 0
    gray_count = 0
    with tqdm(total=total_frames) as pbar:
        pbar.set_description("Extracting all frames")
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count[1] += 1

            if frame_count[1] < 350:
                continue
            # 350 is a sepcial number for the video "leningrad.mp4"
            if rgb_count == gray_count:
                quotient = 75
            else:
                quotient = 175

            if (ret and ((frame_count[1] - frame_count[0]) == quotient)) or frame_count[1] == 350:
                frame_count[0] = frame_count[1]
                if rgb_count == gray_count:
                    # find a rgb frame
                    rgb_count += 1
                    cv2.imwrite(f"imgs/rgb/{rgb_count}.jpg", frame)
                else:
                    # find a gray frame
                    gray_count += 1
                    cv2.imwrite(f"imgs/gray/{gray_count}.jpg", frame)
            if ret and save_all:
                cv2.imwrite(f"video/all_frames/{frame_count[1]}.jpg", frame)
            if gray_count == 50 and not save_all:
                # only need 50 pairs of frames
                break
            pbar.update(1)

            if cv2.waitKey(1)&0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.getcwd())))
    print(f"Current working directory: {os.getcwd()}")
    if not os.path.exists("video/all_frames"):
        os.makedirs("video/all_frames")
    if os.path.exists("imgs/rgb"):    
        shutil.rmtree("imgs/rgb")
    if os.path.exists("imgs/gray"):
        shutil.rmtree("imgs/gray")

    print("Making directory for rgb frames and gray frames")
    os.makedirs("imgs/rgb")
    os.makedirs("imgs/gray")

    extract("video/leningrad.mp4")