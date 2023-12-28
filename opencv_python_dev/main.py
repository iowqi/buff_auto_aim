import numpy as np
import cv2

windows = ("frame", "binary")
for winname in windows:
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, 1280, 720)

RED = 0
BLUE = 1
target_color = RED
oppo_color = ((target_color == RED) * 255, (target_color == RED) * 255, (target_color == BLUE) * 255)

video = None
luminance_threshold = None
saturation_threshold = None
if target_color == RED:
    video = "/home/iowqi/buff_ws/video/main.mp4"
    luminance_threshold = 128
    saturation_threshold = 80
elif target_color == BLUE:
    video = "/home/iowqi/buff_ws/video/main_blue.mp4"
    luminance_threshold = 200
    saturation_threshold = 20

trackerbars = (
    ("luminance_threshold", "binary", luminance_threshold, 255),
    ("morph_close_factor", "binary", 2, 10),
    ("saturation_threshold", "frame", saturation_threshold, 255),
)
for barname, winname, default, maxval in trackerbars:
    cv2.createTrackbar(barname, winname, default, maxval, lambda x: None)

cap = cv2.VideoCapture(video)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    # cv2.imshow("frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lth = cv2.getTrackbarPos(*trackerbars[0][:2])
    _, binmat = cv2.threshold(gray, lth, 255, cv2.THRESH_BINARY)
    mcf = cv2.getTrackbarPos(*trackerbars[1][:2]) * 2 + 1
    binmat = cv2.morphologyEx(binmat, cv2.MORPH_CLOSE, np.ones((mcf, mcf), np.uint8))
    cv2.imshow("binary", binmat)

    contours, _ = cv2.findContours(binmat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    sth = cv2.getTrackbarPos(*trackerbars[2][:2])
    for contour in contours:
        count_color = 0
        for points in contour:
            for point in points:
                is_red = (
                    int(frame[point[1], point[0], 2])
                    - int(frame[point[1], point[0], 0])
                ) > sth
                is_blue = (
                    int(frame[point[1], point[0], 0])
                    - int(frame[point[1], point[0], 2])
                ) > sth
                if (is_red and (target_color == RED)) or (
                    is_blue and (target_color == BLUE)
                ):
                    count_color += 1
                    cv2.circle(frame, (point[0], point[1]), 0, (255, 128, 255), -1)
        count_color /= len(contour)
        # print(count_color)
        if count_color > 0.5:
            #TODO:凸包识别圆形（可以取消形态学变换）再根据两侧灯条特征判断
            cv2.drawContours(frame, [contour], 0, oppo_color, 1)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
