import numpy as np
import cv2 as cv
import glob
from pathlib import Path

images = glob.glob('Faculty_Z_2_frames/*.png')

counter =0
for fname in images:
    if(counter==500):
        break
    print(Path(fname).stem)
    counter=counter+1


