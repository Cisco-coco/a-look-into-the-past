# you should run code in code/ directory

# generate data first
python extract_frame.py

# generate leningrad frames, using yolo to detect persons and saliency map as a complement
python main.py

# generate one past vs one present image where homoGraphy is used to warp the past image
python main.py --homoConvert True

# generate one past vs multi present images where homoGraphy is used to warp the past image
python video_process.py