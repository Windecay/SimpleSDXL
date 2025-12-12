# https://github.com/ssj9596/One-to-All-Animation

import cv2
import numpy as np
import math
import copy

eps = 0.01

DROP_FACE_POINTS = {0, 14, 15, 16, 17}
DROP_UPPER_POINTS = {0, 14, 15, 16, 17, 2, 1, 5, 3, 6}
DROP_LOWER_POINTS = {8, 9, 10, 11, 12, 13}

def scale_and_translate_pose(tgt_pose, ref_pose, conf_th=0.9, return_ratio=False):
    aligned_pose = copy.deepcopy(tgt_pose)
    th = 1e-6
    ref_kpt = ref_pose['bodies']['candidate'].astype(np.float32)
    tgt_kpt = aligned_pose['bodies']['candidate'].astype(np.float32)

    ref_sc = ref_pose['bodies'].get('score', np.ones(ref_kpt.shape[0])).astype(np.float32).reshape(-1)
    tgt_sc = tgt_pose['bodies'].get('score', np.ones(tgt_kpt.shape[0])).astype(np.float32).reshape(-1)

    ref_shoulder_valid = (ref_sc[2] >= conf_th) and (ref_sc[5] >= conf_th)
    tgt_shoulder_valid = (tgt_sc[2] >= conf_th) and (tgt_sc[5] >= conf_th)
    shoulder_ok = ref_shoulder_valid and tgt_shoulder_valid

    ref_hip_valid = (ref_sc[8] >= conf_th) and (ref_sc[11] >= conf_th)
    tgt_hip_valid = (tgt_sc[8] >= conf_th) and (tgt_sc[11] >= conf_th)
    hip_ok = ref_hip_valid and tgt_hip_valid

    if shoulder_ok and hip_ok:
        ref_shoulder_w = abs(ref_kpt[5, 0] - ref_kpt[2, 0])
        tgt_shoulder_w = abs(tgt_kpt[5, 0] - tgt_kpt[2, 0])
        x_ratio = ref_shoulder_w / tgt_shoulder_w if tgt_shoulder_w > th else 1.0

        ref_torso_h = abs(np.mean(ref_kpt[[8, 11], 1]) - np.mean(ref_kpt[[2, 5], 1]))
        tgt_torso_h = abs(np.mean(tgt_kpt[[8, 11], 1]) - np.mean(tgt_kpt[[2, 5], 1]))
        y_ratio = ref_torso_h / tgt_torso_h if tgt_torso_h > th else 1.0
        scale_ratio = (x_ratio + y_ratio) / 2

    elif shoulder_ok:
        ref_sh_dist = np.linalg.norm(ref_kpt[2] - ref_kpt[5])
        tgt_sh_dist = np.linalg.norm(tgt_kpt[2] - tgt_kpt[5])
        scale_ratio = ref_sh_dist / tgt_sh_dist if tgt_sh_dist > th else 1.0

    else:
        ref_ear_dist = np.linalg.norm(ref_kpt[16] - ref_kpt[17])
        tgt_ear_dist = np.linalg.norm(tgt_kpt[16] - tgt_kpt[17])
        scale_ratio = ref_ear_dist / tgt_ear_dist if tgt_ear_dist > th else 1.0

    if return_ratio:
        return scale_ratio

    # scale
    anchor_idx = 1
    anchor_pt_before_scale = tgt_kpt[anchor_idx].copy()
    def scale(arr):
        if arr is not None and arr.size > 0:
            arr[..., 0] = anchor_pt_before_scale[0] + (arr[..., 0] - anchor_pt_before_scale[0]) * scale_ratio
            arr[..., 1] = anchor_pt_before_scale[1] + (arr[..., 1] - anchor_pt_before_scale[1]) * scale_ratio
    scale(tgt_kpt)
    scale(aligned_pose.get('faces'))
    scale(aligned_pose.get('hands'))

    # offset
    offset = ref_kpt[anchor_idx] - tgt_kpt[anchor_idx]
    def translate(arr):
        if arr is not None and arr.size > 0:
            arr += offset
    translate(tgt_kpt)
    translate(aligned_pose.get('faces'))
    translate(aligned_pose.get('hands'))
    aligned_pose['bodies']['candidate'] = tgt_kpt

    return aligned_pose, shoulder_ok, hip_ok


def warp_ref_to_pose(tgt_img,
                     ref_pose: dict, #driven pose
                     tgt_pose: dict,
                     bg_val=(0, 0, 0),
                     conf_th=0.9,
                     align_center=False):

    H, W = tgt_img.shape[:2]
    img_tgt_pose = draw_pose_aligned(tgt_pose, H, W, without_face=True)

    tgt_kpt = tgt_pose['bodies']['candidate'].astype(np.float32)
    ref_kpt = ref_pose['bodies']['candidate'].astype(np.float32)

    scale_ratio = scale_and_translate_pose(tgt_pose, ref_pose, conf_th=conf_th, return_ratio=True)

    anchor_idx = 1
    x0 = tgt_kpt[anchor_idx][0] * W
    y0 = tgt_kpt[anchor_idx][1] * H

    ref_x = ref_kpt[anchor_idx][0] * W if not align_center else W/2
    ref_y = ref_kpt[anchor_idx][1] * H

    dx = ref_x - x0
    dy = ref_y - y0

    # Affine transformation matrix
    M = np.array([[scale_ratio, 0, (1-scale_ratio)*x0 + dx],
                  [0, scale_ratio, (1-scale_ratio)*y0 + dy]],
                 dtype=np.float32)
    img_warp = cv2.warpAffine(tgt_img, M, (W, H),
                              flags=cv2.INTER_LINEAR,
                              borderValue=bg_val)
    img_tgt_pose_warp = cv2.warpAffine(img_tgt_pose, M, (W, H),
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=bg_val)
    zeros = np.zeros((H, W), dtype=np.uint8)
    mask_warp = cv2.warpAffine(zeros, M, (W, H),
                               flags=cv2.INTER_NEAREST,
                               borderValue=255)
    return img_warp, img_tgt_pose_warp, mask_warp

def hsv_to_rgb(hsv):
    hsv = np.asarray(hsv, dtype=np.float32)
    in_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)

    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    i = i % 6

    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = np.zeros_like(hsv)
    rgb[i == 0] = np.stack([v[i == 0], t[i == 0], p[i == 0]], axis=1)
    rgb[i == 1] = np.stack([q[i == 1], v[i == 1], p[i == 1]], axis=1)
    rgb[i == 2] = np.stack([p[i == 2], v[i == 2], t[i == 2]], axis=1)
    rgb[i == 3] = np.stack([p[i == 3], q[i == 3], v[i == 3]], axis=1)
    rgb[i == 4] = np.stack([t[i == 4], p[i == 4], v[i == 4]], axis=1)
    rgb[i == 5] = np.stack([v[i == 5], p[i == 5], q[i == 5]], axis=1)

    gray_mask = s == 0
    rgb[gray_mask] = np.stack([v[gray_mask]] * 3, axis=1)

    return (rgb.reshape(in_shape) * 255)

def get_stickwidth(W, H, stickwidth=4):
    if max(W, H) < 512:
        ratio = 1.0
    elif max(W, H) < 1080:
        ratio = 1.5
    elif max(W, H) < 2160:
        ratio = 2.0
    elif max(W, H) < 3240:
        ratio = 2.5
    elif max(W, H) < 4320:
        ratio = 3.5
    elif max(W, H) < 5400:
        ratio = 4.5
    else:
        ratio = 4.0
    return int(stickwidth * ratio)


def alpha_blend_color(color, alpha):
    return [int(c * alpha) for c in color]


def draw_bodypose_aligned(canvas, candidate, subset, score, plan=None):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    stickwidth = get_stickwidth(W, H, stickwidth=3)

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
        [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    HIDE_JOINTS = set()
    stretch_limb_idx = None
    stretch_scale = None
    if plan:
        if plan["mode"] == "drop_point":
            HIDE_JOINTS.add(plan["point_idx"])
        elif plan["mode"] == "drop_region":
            HIDE_JOINTS |= set(plan["points"])
        elif plan["mode"] == "stretch_limb":
            stretch_limb_idx = plan["limb_idx"]
            stretch_scale = plan["stretch_scale"]

    hide_joint = np.zeros_like(subset, dtype=bool)

    for i in range(17):
        for n in range(len(subset)):
            idx_pair = limbSeq[i]

            if any(j in HIDE_JOINTS for j in idx_pair):
                continue

            index = subset[n][np.array(idx_pair) - 1]
            conf = score[n][np.array(idx_pair) - 1]
            if -1 in index:
                continue
            # color lighten
            alpha = max(conf[0] * conf[1], 0) if conf[0]>0 and conf[1]>0 else 0.35
            if conf[0] == 0 or conf[1] == 0:
                alpha = 0

            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)

            if stretch_limb_idx == i:
                vec_x = X[1] - X[0]
                vec_y = Y[1] - Y[0]
                X[1] = X[0] + vec_x * stretch_scale
                Y[1] = Y[0] + vec_y * stretch_scale
                hide_joint[n, idx_pair[1]-1] = True

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0]-X[1])**2 + (Y[0]-Y[1])**2) ** 0.5
            angle = math.degrees(math.atan2(X[0]-X[1], Y[0]-Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], alpha))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        if i in HIDE_JOINTS:
            continue
        for n in range(len(subset)):
            if hide_joint[n, i]:
                continue
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]

            alpha = 0 if conf==-2 else max(conf, 0)
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (x, y), stickwidth, alpha_blend_color(colors[i], alpha), thickness=-1)

    return canvas


def draw_handpose_aligned(canvas, all_hand_peaks, all_hand_scores, draw_th=0.3):
    H, W, C = canvas.shape
    stickwidth = get_stickwidth(W, H, stickwidth=2)
    line_thickness = get_stickwidth(W, H, stickwidth=2)

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        for ie, e in enumerate(edges):
            if scores[e[0]] < draw_th or scores[e[1]] < draw_th:
                    continue
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)

            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                color = hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]).flatten()
                color = tuple(int(c * score / 255) for c in color)
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=line_thickness)

        for i, keyponit in enumerate(peaks):
            if scores[i] < draw_th:
                continue

            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), stickwidth, (0, 0, score), thickness=-1)
    return canvas


def draw_facepose_aligned(canvas, all_lmks, all_scores, draw_th=0.3,face_change=False):
    H, W, C = canvas.shape
    stickwidth = get_stickwidth(W, H, stickwidth=2)
    SKIP_IDX = set(range(0, 17))
    SKIP_IDX |= set(range(27, 36))

    for lmks, scores in zip(all_lmks, all_scores):
        for idx, (lmk, score) in enumerate(zip(lmks, scores)):
            # skip chin
            if idx in SKIP_IDX:
                continue
            if score < draw_th:
                continue
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = int(score * 255)
            # color lighten
            if face_change:
                conf = int(conf * 0.35)

            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), stickwidth, (conf, conf, conf), thickness=-1)
    return canvas


def draw_pose_aligned(pose, H, W, ref_w=2160, without_face=False, pose_plan=None, head_strength="full", face_change=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    body_score = bodies['score'].copy()
    # control color
    if head_strength == "weak":
        target_joints = [0, 14, 15, 16, 17]
        body_score[:, target_joints] = -2
    elif head_strength == "none":
        target_joints = [0, 14, 15, 16, 17]
        body_score[:, target_joints] = 0

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    canvas = draw_bodypose_aligned(canvas, candidate, subset,
                                   score=body_score,
                                   plan=pose_plan,)

    canvas = draw_handpose_aligned(canvas, hands, pose['hands_score'])

    if not without_face:
        canvas = draw_facepose_aligned(canvas, faces, pose['faces_score'],face_change=face_change)

    return cv2.resize(canvas, (W, H))
