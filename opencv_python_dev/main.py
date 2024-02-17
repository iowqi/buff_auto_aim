import numpy as np
import cv2
import pyclipper
import copy

windows = ("frame", "binary", "result")
for winname in windows:
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, 1280, 720)

RED = 0
BLUE = 1
target_color = 1
oppo_color = (
    (target_color == RED) * 255,
    (target_color == RED) * 200,
    (target_color == BLUE) * 255,
)


paused = False
video = None
luminance_threshold = None
saturation_threshold = None
if target_color == RED:
    video = "/home/iowqi/buff_ws/video/main.mp4"
    luminance_threshold = 80
    saturation_threshold = 80
elif target_color == BLUE:
    video = "/home/iowqi/buff_ws/video/main_blue.mp4"
    luminance_threshold = 200
    saturation_threshold = 128

trackerbars = (
    ("luminance_threshold", "binary", luminance_threshold, 255),
    ("close_factor", "binary", 0, 10),
    ("saturation_threshold", "binary", saturation_threshold, 255),
    ("color_point_ratio", "binary", 50, 100),
    ("square_like_ratio", "result", 90, 100),
)
for barname, winname, default, maxval in trackerbars:
    cv2.createTrackbar(barname, winname, default, maxval, lambda x: None)


def equidistant_zoom_contour(contour, margin):
    """
    等距离缩放多边形轮廓点
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param margin: 轮廓外扩的像素距离,margin正数是外扩,负数是缩小
    :return: 外扩后的轮廓点
    """
    pco = pyclipper.PyclipperOffset()
    ##### 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
    pco.MiterLimit = 10
    contour = contour[:, 0, :]
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    # print(solution)
    if len(solution) > 1:
        for i in range(1, len(solution)):
            solution[0] = np.concatenate((solution[0], solution[i]), axis=0)
        solution = [solution[0]]
    solution = np.array(solution).reshape(-1, 1, 2).astype(int)
    return solution


def normalize_img(img):
    """
    归一化图像
    :param img:
    :return:
    """
    img = cv2.resize(img, (64, 64))
    return img


target_tamplate = normalize_img(cv2.imread("target.png", cv2.IMREAD_GRAYSCALE))
medium_ring_tamplate = normalize_img(
    cv2.imread("medium_ring.png", cv2.IMREAD_GRAYSCALE)
)
r_logo_tamplate = normalize_img(cv2.imread("R_logo.png", cv2.IMREAD_GRAYSCALE))
tamplates = [target_tamplate, medium_ring_tamplate, r_logo_tamplate]

cap = cv2.VideoCapture(video)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    cv2.imshow("frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lth = cv2.getTrackbarPos(*trackerbars[0][:2])
    cv2.GaussianBlur(gray, (5, 5), 0)
    _, binmat = cv2.threshold(gray, lth, 255, cv2.THRESH_BINARY)
    mcf = cv2.getTrackbarPos(*trackerbars[1][:2]) * 2 + 1
    binmat_morpho = cv2.morphologyEx(
        binmat, cv2.MORPH_CLOSE, np.ones((mcf, mcf), np.uint8)
    )

    merge_binmat = cv2.merge([gray, cv2.bitwise_xor(binmat, binmat_morpho), binmat])

    contours, _ = cv2.findContours(
        binmat_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    sth = cv2.getTrackbarPos(*trackerbars[2][:2])  # saturation_threshold
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue

        offset = equidistant_zoom_contour(np.array(contour), 2)

        count_color = 0
        for points in offset:
            for point in points:
                if (
                    point[0] < 0
                    or point[0] >= frame.shape[1]
                    or point[1] < 0
                    or point[1] >= frame.shape[0]
                ):
                    continue
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
                    cv2.circle(merge_binmat, (point[0], point[1]), 0, (0, 255, 0), -1)
        count_color /= len(offset)
        # print(count_color)

        cpr = cv2.getTrackbarPos(*trackerbars[3][:2]) / 100  # color_point_ratio
        if count_color > cpr:
            cv2.drawContours(merge_binmat, [contour], 0, (255, 200, 0), 1)

            rect = cv2.minAreaRect(contour)

            slr = cv2.getTrackbarPos(*trackerbars[4][:2]) / 100  # square_like_ratio
            is_square = 1 - abs(rect[1][0] - rect[1][1]) / rect[1][0] > slr
            if not is_square:  # 正方形过滤
                continue

            box = cv2.boxPoints(rect)  # 绘制包裹矩形
            box = np.intp(box)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 1)

            # TODO:凸包识别圆形（可以取消形态学变换）再根据两侧灯条特征判断是靶心圆还是环数圆，非圆通过神经网络判断R标相似度

            hull = cv2.convexHull(contour)  # 获得经过凸包掩膜处理后的ROI
            roi = cv2.boundingRect(hull)
            roi_img = gray[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [hull], 0, 255, -1)
            roi_img = cv2.bitwise_and(
                roi_img, mask[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            )
            roi_img = normalize_img(roi_img)  # 归一化大小
            cv2.imshow("roi", roi_img)

            # 计算匹配得分
            scores = []
            for tamplate in tamplates:  # 在模板中计算每一个得分
                result = cv2.matchTemplate(roi_img, tamplate, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            max_score_arg = np.argmax(scores)

            mu = cv2.moments(hull)  # 获得图像矩
            cx = mu["m10"] / mu["m00"]  # 表示列数，是横坐标
            cy = mu["m01"] / mu["m00"]

            shift = 3  # 绘制目标中心(抗锯齿圆形)
            factor = 1 << shift
            cv2.circle(
                frame,
                (int(cx * factor + 0.5), int(cy * factor + 0.5)),
                10 * factor,
                (1, 0, 0),
                1,
                cv2.LINE_AA,
                shift,
            )

            cv2.polylines(frame, [hull], True, oppo_color, 1)

            type_name = None  # 绘制模板匹配结果
            if max_score_arg == 0:
                type_name = "T"
                cv2.imshow("T", roi_img)
            elif max_score_arg == 1:
                type_name = "C"
                cv2.imshow("C", roi_img)
            else:
                type_name = "R"
                cv2.imshow("R", roi_img)
            cv2.putText(
                frame,
                "%s:%.1f" % (type_name, scores[max_score_arg] / (10**6)),
                np.intp([cx, cy]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                oppo_color,
                1,
            )

    cv2.imshow("binary", merge_binmat)
    cv2.imshow("result", frame)

    key = cv2.waitKey(int(not paused))
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord(" "):
        paused = not paused
cap.release()
cv2.destroyAllWindows()
