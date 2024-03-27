import numpy as np
import cv2
import pyclipper
from scipy.spatial import KDTree
import time

WIN_NAMES = ("frame", "binary", "result")
ROI_WIN_NAMES = ("target", "ring", "rlogo")
for WINNAME in WIN_NAMES:
    cv2.namedWindow(WINNAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINNAME, 1280, 720)

ROOT_PATH="./src/buff_auto_aim/opencv_python_dev/"

RED = 0
BLUE = 1
TARGET_MAIN = 1
COLOR_OPPO = ((TARGET_MAIN == RED) * 255, (TARGET_MAIN == RED) * 200, (TARGET_MAIN == BLUE) * 255,)
COLOR_GREEN = (0, 255, 0)
COLOR_GRAY = (100, 100, 100)
TARGET_OBJ_COLORS = [COLOR_OPPO, (128, 128, 128), (0, 255, 0)]

VIDEO = None
LUMINANCE_TH = None
SATURATE_TH = None
if TARGET_MAIN == RED:
    VIDEO = "./video/main.mp4"
    LUMINANCE_TH = 80
    SATURATE_TH = 80
elif TARGET_MAIN == BLUE:
    VIDEO = "./video/main_blue.mp4"
    LUMINANCE_TH = 200
    SATURATE_TH = 128

trackerbars = (
    ("luminance_threshold", "binary", LUMINANCE_TH, 255),
    ("close_factor", "binary", 0, 10),
    ("saturation_threshold", "binary", SATURATE_TH, 255),
    ("color_point_ratio", "binary", 50, 100),
    ("square_like_ratio", "result", 90, 100),
    ("normalize_resolution", "result", 16, 256),
    ("minimum_confidence_level", "result", 60, 100),
)
for barname, WINNAME, default, maxval in trackerbars:
    cv2.createTrackbar(barname, WINNAME, default, maxval, lambda x: None)

TAMPLATE_PATHS = ["target.png", "medium_ring.png", "R_logo.png"]
NONNORMALIZED_T = [cv2.imread(ROOT_PATH+fp, cv2.IMREAD_GRAYSCALE) for fp in TAMPLATE_PATHS]


def draw_AA_circle(img, center, color, radius=5, thickness=1):
    """
    绘制抗锯齿圆形
    """
    shift = 3  # 绘制目标中心(抗锯齿圆形)
    factor = 1 << shift
    cv2.circle(img, (int(center[0] * factor + 0.5), int(center[1] * factor + 0.5)),
               radius * factor, color, thickness, cv2.LINE_AA, shift,)


def equidistant_zoom_contour(contour, margin):
    """
    等距离缩放多边形轮廓点
    :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
    :param margin: 轮廓外扩的像素距离,margin正数是外扩,负数是缩小
    :return: 外扩后的轮廓点
    """
    pco = pyclipper.PyclipperOffset()
    # 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
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


def normalize_img(img, dsize):
    """
    归一化图像
    :param img:
    :return:
    """
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
    return img


paused = False
start_time = time.time()
cap = cv2.VideoCapture(VIDEO)
while cap.isOpened():
    # 更新滑条值
    lth, mcf, sth, cpr, slr, nre, mcl = [cv2.getTrackbarPos(*trackerbar[:2]) for trackerbar in trackerbars]
    mcf, cpr, slr, nre, mcl = mcf * 2 + 1, cpr / 100, slr / 100, (max(nre, 1), max(nre, 1)), mcl / 100
    tamplates = [normalize_img(t, nre) for t in NONNORMALIZED_T]

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    cv2.imshow("frame", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 亮度阈值处理
    _, binmat = cv2.threshold(gray, lth, 255, cv2.THRESH_BINARY)

    binmat_morpho = cv2.morphologyEx(binmat, cv2.MORPH_CLOSE, np.ones((mcf, mcf), np.uint8))  # 形态学处理
    merge_binmat = cv2.merge([gray, cv2.bitwise_xor(binmat, binmat_morpho), binmat])

    contours, _ = cv2.findContours(binmat_morpho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓处理

    non_target = []  # [((cx,cy),(w,h),a),()...]
    main_objs = [target_objs, ring_objs, rlogo_objs] = [[], [], []]  # each of [((cx,cy,size),(3 of scores)),()...]
    for contour in contours:
        if cv2.contourArea(contour) < 80:  # 过滤小面积
            continue

        offset_contour = equidistant_zoom_contour(np.array(contour), 2)  # 获得偏移后的轮廓
        color_counter = 0
        for points in offset_contour:  # 统计符合目标颜色的点数
            for point in points:
                if (point[0] < 0 or point[0] >= frame.shape[1] or point[1] < 0 or point[1] >= frame.shape[0]):
                    continue  # 过滤屏幕外的点
                is_red = (int(frame[point[1], point[0], 2]) - int(frame[point[1], point[0], 0])) > sth
                is_blue = (int(frame[point[1], point[0], 0]) - int(frame[point[1], point[0], 2])) > sth
                if (is_red and (TARGET_MAIN == RED)) or (is_blue and (TARGET_MAIN == BLUE)):
                    color_counter += 1
                    cv2.circle(merge_binmat, (point[0], point[1]), 0, (0, 255, 0), -1)
        color_counter /= len(offset_contour)
        if color_counter < cpr:  # 过滤非目标颜色的轮廓
            continue

        cv2.drawContours(merge_binmat, [contour], 0, (255, 200, 0), 1)

        rect = cv2.minAreaRect(contour)
        is_square = 1 - abs(rect[1][0] - rect[1][1]) / rect[1][0] > slr
        if not is_square:  # 过滤非正方形轮廓
            non_target.append(rect)
            continue

        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(frame, [box], 0, (128, 128, 128), 1)

        # TODO:凸包识别圆形（可以取消形态学变换）再根据两侧灯条特征判断是靶心圆还是环数圆，非圆通过神经网络判断R标相似度

        hull = cv2.convexHull(contour)  # 获得经过凸包掩膜处理后的ROI
        roi = cv2.boundingRect(hull)
        roi_img = gray[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]]
        mask = np.zeros_like(gray)  # 从凸包创建掩膜
        cv2.drawContours(mask, [hull], 0, 255, -1)
        roi_img_masked = cv2.bitwise_and(roi_img, mask[roi[1]: roi[1] + roi[3], roi[0]: roi[0] + roi[2]])
        normalized_roi_img = normalize_img(roi_img_masked, nre)  # 归一化大小

        # 计算匹配得分
        scores = []
        for tamplate in tamplates:  # 在模板中计算每一个得分
            res = cv2.matchTemplate(normalized_roi_img, tamplate, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            scores.append(max_val)
        target_arg = np.argmax(scores)
        if scores[target_arg] < mcl:  # 过滤完全和模板不相同的轮廓
            # non_target.append(rect)
            continue

        mu = cv2.moments(hull)  # 获得轮廓矩
        cx = mu["m10"] / mu["m00"]  # 表示列数，是横坐标
        cy = mu["m01"] / mu["m00"]
        main_objs[target_arg].append(((cx, cy, np.mean(rect[1][:2])), scores[:3]))

        # 绘制模板匹配结果
        cv2.imshow(ROI_WIN_NAMES[target_arg], roi_img_masked)
        cv2.polylines(frame, [hull], True, COLOR_OPPO, 1)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 1)
        cv2.putText(frame, "%dT %dC %dR" % ((scores[0]) * 100, (scores[1]) * 100, (scores[2]) * 100,),
                    np.intp([cx + int(rect[1][0] / 2), cy - int(rect[1][1] / 2)]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TARGET_OBJ_COLORS[target_arg], 1, cv2.LINE_AA)

    if len(rlogo_objs):
        rlogo_obj = max(rlogo_objs, key=lambda x: x[1][2])[0]
        draw_AA_circle(frame, rlogo_obj[:2], TARGET_OBJ_COLORS[2])
        if len(non_target) and len(target_objs):
            points_set = np.array([t[0] for t in non_target])
            non_target_tree = KDTree(points_set)  # 建立KD搜索树
            target_obj = max(target_objs, key=lambda x: x[1][0])[0]
            nearest_point = non_target_tree.query(target_obj[:2], k=1)[1]
            draw_AA_circle(frame, non_target[nearest_point][0][:2], TARGET_OBJ_COLORS[0], 20, 2)
            box = cv2.boxPoints(non_target[nearest_point])
            box = np.intp(box)
            cv2.drawContours(frame, [box], 0, (0,0,255), 1)

    for it in enumerate(ROI_WIN_NAMES):
        cv2.putText(frame, "%s: %d" % (it[1], len(main_objs[it[0]])), (10, 30 + (it[0]+1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TARGET_OBJ_COLORS[it[0]], 1, cv2.LINE_AA)
    cv2.putText(frame, "non_target: %d" % len(non_target), (10, 30 + 4 * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "FPS: %.2f" % (1.0 / (time.time() - start_time)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    start_time = time.time()
    cv2.imshow("binary", merge_binmat)
    cv2.imshow("result", frame)

    key = cv2.waitKey(int(not paused))
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord(" "):
        paused = not paused
cap.release()
cv2.destroyAllWindows()
