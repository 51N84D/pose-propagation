import cv2
from pathlib import Path

video_path = Path(
    "/Users/Sunsmeister/Desktop/Research/Brain/DGP/dgp_propagate/data/reach/raw_data/reachingvideo1.avi"
)

write_path = Path("./video_frames")
write_path.mkdir(exist_ok=True)
print("write path: ", write_path)

vc = cv2.VideoCapture(str(video_path))
c = 1
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    cv2.imwrite(str(write_path / Path(str(c) + ".jpg")), frame)
    c = c + 1
    cv2.waitKey(1)
vc.release()
