import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return


save_all_frames('my_vid/Faculty_long_2.mp4', 'Faculty_long_2_frames', '', 'png')