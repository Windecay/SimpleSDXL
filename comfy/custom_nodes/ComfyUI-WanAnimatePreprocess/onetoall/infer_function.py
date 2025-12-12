# https://github.com/ssj9596/One-to-All-Animation

import numpy as np
import copy
from ..retarget_pose import get_retarget_pose

L_EYE_IDXS = list(range(36, 42))
R_EYE_IDXS = list(range(42, 48))
NOSE_TIP = 30
MOUTH_L = 48
MOUTH_R = 54
JAW_LINE = list(range(0, 17))


# ===========================Convert wanpose format into our dwpose-like format======================
def aaposemeta_to_dwpose(meta):
    candidate_body = meta['keypoints_body'][:-2][:, :2]
    score_body = meta['keypoints_body'][:-2][:, 2]
    subset_body = np.arange(len(candidate_body), dtype=float)
    subset_body[score_body <= 0] = -1
    bodies = {
        "candidate": candidate_body,
        "subset": np.expand_dims(subset_body, axis=0),   # shape (1, N)
        "score": np.expand_dims(score_body, axis=0)      # shape (1, N)
    }
    hands_coords = np.stack([
        meta['keypoints_right_hand'][:, :2],
        meta['keypoints_left_hand'][:, :2]
    ])
    hands_score = np.stack([
        meta['keypoints_right_hand'][:, 2],
        meta['keypoints_left_hand'][:, 2]
    ])
    faces_coords = np.expand_dims(meta['keypoints_face'][1:][:, :2], axis=0)
    faces_score = np.expand_dims(meta['keypoints_face'][1:][:, 2], axis=0)
    dwpose_format = {
        "bodies": bodies,
        "hands": hands_coords,
        "hands_score": hands_score,
        "faces": faces_coords,
        "faces_score": faces_score
    }
    return dwpose_format

def aaposemeta_obj_to_dwpose(pose_meta):
    """
    Convert an AAPoseMeta object into a dwpose-like data structure
    Restore coordinates to relative coordinates (divide by width, height)
    Only handle None -> fill with zeros
    """
    w = pose_meta.width
    h = pose_meta.height

    # If None, fill with all zeros
    def safe(arr, like_shape):
        if arr is None:
            return np.zeros(like_shape, dtype=np.float32)
        arr_np = np.array(arr, dtype=np.float32)
        arr_np = np.nan_to_num(arr_np, nan=0.0)
        return arr_np
    # body
    kps_body = safe(pose_meta.kps_body, (pose_meta.kps_body_p.shape[0], 2))
    candidate_body = kps_body / np.array([w, h])
    score_body = safe(pose_meta.kps_body_p, (candidate_body.shape[0],))
    subset_body = np.arange(len(candidate_body), dtype=float)
    subset_body[score_body <= 0] = -1
    bodies = {
        "candidate": candidate_body,
        "subset": np.expand_dims(subset_body, axis=0),
        "score": np.expand_dims(score_body, axis=0)
    }

    # hands
    kps_rhand = safe(pose_meta.kps_rhand, (pose_meta.kps_rhand_p.shape[0], 2))
    kps_lhand = safe(pose_meta.kps_lhand, (pose_meta.kps_lhand_p.shape[0], 2))
    hands_coords = np.stack([
        kps_rhand / np.array([w, h]),
        kps_lhand / np.array([w, h])
    ])
    hands_score = np.stack([
        safe(pose_meta.kps_rhand_p, (kps_rhand.shape[0],)),
        safe(pose_meta.kps_lhand_p, (kps_lhand.shape[0],))
    ])

    dwpose_format = {
        "bodies": bodies,
        "hands": hands_coords,
        "hands_score": hands_score,
        "faces": None,
        "faces_score": None
    }
    return dwpose_format

# ===============================Face Rough alignment======================

def _to_68x2(arr):
    if arr.shape == (1, 68, 2):
        def to_orig(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (68, 2):
                raise ValueError("to_orig expects (68,2)")
            return x[np.newaxis, :, :]
        return arr[0].astype(np.float64), to_orig
    if arr.shape == (68, 2):
        def to_orig(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (68, 2):
                raise ValueError("to_orig expects (68,2)")
            return x
        return arr.astype(np.float64), to_orig
    if arr.shape == (2, 68):
        def to_orig(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (68, 2):
                raise ValueError("to_orig expects (68,2)")
            return x.T
        return arr.T.astype(np.float64), to_orig
    raise ValueError(f"faces shape {arr.shape} not supported; expected (1,68,2) or (68,2) or (2,68)")

def _eye_center(face68, idxs):
    return face68[idxs].mean(axis=0)

def _anchors(face68):
    le = _eye_center(face68, L_EYE_IDXS)
    re = _eye_center(face68, R_EYE_IDXS)
    nose = face68[NOSE_TIP]
    lm = face68[MOUTH_L]
    rm = face68[MOUTH_R]
    if re[0] < le[0]:
        le, re = re, le
    return np.stack([le, re, nose, lm, rm], axis=0)

def _face_scale_only(src68, ref68, target_nose_pos, alpha=1.0, anchor_pairs=[[36, 45], [27, 8]]):
    """
    Rough alignment - adjust the shape of the source face according to the proportions of the reference, and align the nose tip to target_nose_pos.
    anchor_pairs:
      - [36, 45] for x
      - [27, 8] for y
    """
    src = np.asarray(src68, dtype=np.float64)
    ref = np.asarray(ref68, dtype=np.float64)

    center = _anchors(src).mean(axis=0)
    src_centered = src - center

    src_w = np.linalg.norm(src[anchor_pairs[0][0]] - src[anchor_pairs[0][1]])
    ref_w = np.linalg.norm(ref[anchor_pairs[0][0]] - ref[anchor_pairs[0][1]])

    src_h = np.linalg.norm(src[anchor_pairs[1][0]] - src[anchor_pairs[1][1]])
    ref_h = np.linalg.norm(ref[anchor_pairs[1][0]] - ref[anchor_pairs[1][1]])

    scale_x = ref_w / src_w if src_w > 1e-6 else 1.0
    scale_y = ref_h / src_h if src_h > 1e-6 else 1.0

    scaled_local = src_centered.copy()
    scaled_local[:, 0] *= (1 - alpha) + scale_x * alpha
    scaled_local[:, 1] *= (1 - alpha) + scale_y * alpha
    scaled_global = scaled_local + center

    nose_idx = NOSE_TIP
    current_nose = scaled_global[nose_idx]
    offset = target_nose_pos - current_nose
    scaled_global += offset

    return scaled_global

# ===============================Reference Img Pre-Process======================


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

# ===============================Align to Ref Driven Pose Retarget ======================

def align_to_reference(ref_pose_meta, tpl_pose_metas, tpl_dwposes, anchor_idx=None):
    # pose retarget + face rough align

    ref_pose_dw = aaposemeta_to_dwpose(ref_pose_meta)
    best_idx = anchor_idx
    tpl_pose_meta_best = tpl_pose_metas[best_idx]

    tpl_retarget_pose_metas = get_retarget_pose(
        tpl_pose_meta_best,
        ref_pose_meta,
        tpl_pose_metas,
        None, None
    )

    retarget_dwposes = [aaposemeta_obj_to_dwpose(pm) for pm in tpl_retarget_pose_metas]

    if ref_pose_dw['faces'] is not None:
        ref68, _ = _to_68x2(ref_pose_dw['faces'])
        for frame_idx, (tpl_dw, rt_dw) in enumerate(zip(tpl_dwposes, retarget_dwposes)):
            if tpl_dw['faces'] is None:
                continue
            src68, to_orig = _to_68x2(tpl_dw['faces'])
            target_nose_pos = rt_dw['bodies']['candidate'][0]
            scaled68 = _face_scale_only(src68, ref68, target_nose_pos, alpha=1.0)
            rt_dw['faces'] = to_orig(scaled68)
            rt_dw['faces_score'] = tpl_dw['faces_score']

    return retarget_dwposes

# ===============================Rescale-Ref && Change part of pose(Option)======================


def compute_ratios_stepwise(ref_scores, source_scores, ref_pts, src_pts, conf_th=0.9, th=1e-6):

    def keypoint_valid(idx):
        return ref_scores[0, idx] >= conf_th and source_scores[0, idx] >= conf_th

    def safe_ratio(p1, p2):
        len_ref = np.linalg.norm(ref_pts[p1] - ref_pts[p2])
        len_src = np.linalg.norm(src_pts[p1] - src_pts[p2])
        if len_src > th:
            return len_ref / len_src
        else:
            return 1.0

    ratio_pairs = [
        (0,1),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
        (0,14),(0,15),(14,16),(15,17),
        (8,9),(9,10),(11,12),(12,13),
        (1,8),(1,11)
    ]
    ratios = {p: 1.0 for p in ratio_pairs}

    parent_map = {
        (3, 4): (2, 3),
        (6, 7): (5, 6),
        (9, 10): (8, 9),
        (12, 13): (11, 12)
    }

    # Group 1 — head only
    if all(keypoint_valid(i) for i in [0,1,14,15,16,17]):
        ratios[(0,1)]  = safe_ratio(0,1)
        ratios[(0,14)] = safe_ratio(0,14)
        ratios[(0,15)] = safe_ratio(0,15)
        ratios[(14,16)]= safe_ratio(14,16)
        ratios[(15,17)]= safe_ratio(15,17)

    # Group 2 — +shoulder
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17]):
        ratios[(1,2)] = safe_ratio(1,2)
        ratios[(1,5)] = safe_ratio(1,5)

    # Group 3 — +upper arm
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6]):
        ratios[(2,3)] = safe_ratio(2,3)
        ratios[(5,6)] = safe_ratio(5,6)
        ratios[(3,4)] = ratios[parent_map[(3,4)]]
        ratios[(6,7)] = ratios[parent_map[(6,7)]]

    # Group 4 — +hips
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6,8,11]):
        ratios[(1,8)] = safe_ratio(1,8)
        ratios[(1,11)] = safe_ratio(1,11)

    # Group 5 — forearm own
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6,8,11,4,7]):
        ratios[(3,4)] = safe_ratio(3,4)
        ratios[(6,7)] = safe_ratio(6,7)

    # Group 6 — knees
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6,8,11,4,7,9,12]):
        ratios[(8,9)] = safe_ratio(8,9)
        ratios[(11,12)] = safe_ratio(11,12)
        ratios[(9,10)] = ratios[parent_map[(9,10)]]
        ratios[(12,13)]= ratios[parent_map[(12,13)]]

    # Full body — all ratios
    if all(keypoint_valid(i) for i in range(18)):
        for p in ratio_pairs:
            ratios[p] = safe_ratio(*p)

    symmetric_pairs = [
        ((1, 2), (1, 5)),    # 两肩
        ((2, 3), (5, 6)),    # 上臂
        ((3, 4), (6, 7)),    # 前臂
        ((8, 9), (11, 12)),  # 大腿
        ((9, 10), (12, 13))  # 小腿
    ]
    for left_key, right_key in symmetric_pairs:
        left_val = ratios.get(left_key)
        right_val = ratios.get(right_key)
        if left_val is not None and right_val is not None:
            avg_val = (left_val + right_val) / 2.0
            ratios[left_key] = avg_val
            ratios[right_key] = avg_val

    eye_pairs = [
        ((13, 15), (14, 16))
    ]
    for left_key, right_key in eye_pairs:
        left_val = ratios.get(left_key)
        right_val = ratios.get(right_key)
        if left_val is not None and right_val is not None:
            avg_val = (left_val + right_val) / 2.0
            ratios[left_key] = avg_val
            ratios[right_key] = avg_val

    return ratios

def align_to_pose(ref_dwpose, tpl_dwposes,anchor_idx=None,conf_th=0.9,):
    detected_poses = copy.deepcopy(tpl_dwposes)

    best_pose = tpl_dwposes[anchor_idx]
    ref_pose_scaled, _, _ = scale_and_translate_pose(ref_dwpose, best_pose, conf_th=conf_th)

    ref_candidate = ref_pose_scaled['bodies']['candidate'].astype(np.float32)
    ref_scores    = ref_pose_scaled['bodies']['score'].astype(np.float32)

    source_candidate = best_pose['bodies']['candidate'].astype(np.float32)
    source_scores = best_pose['bodies']['score'].astype(np.float32)

    has_ref_face = 'faces' in ref_pose_scaled and ref_pose_scaled['faces'] is not None and ref_pose_scaled['faces'].size > 0
    if has_ref_face:
        try:
            ref68, _ = _to_68x2(ref_pose_scaled['faces'])
        except Exception as e:
            print("Reference face conversion failed:", e)
            has_ref_face = False

    ratios = compute_ratios_stepwise(ref_scores, source_scores, ref_candidate, source_candidate, conf_th=conf_th, th=1e-6)

    for pose in detected_poses:
        candidate = pose['bodies']['candidate']
        hands = pose['hands']

        # ===== Neck =====
        ratio = ratios[(0, 1)]
        x_offset = (candidate[1][0] - candidate[0][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[0][1]) * (1. - ratio)
        candidate[[0, 14, 15, 16, 17], 0] += x_offset
        candidate[[0, 14, 15, 16, 17], 1] += y_offset

        # ===== Shoulder Right =====
        ratio = ratios[(1, 2)]
        x_offset = (candidate[1][0] - candidate[2][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[2][1]) * (1. - ratio)
        candidate[[2, 3, 4], 0] += x_offset
        candidate[[2, 3, 4], 1] += y_offset
        hands[1, :, 0] += x_offset
        hands[1, :, 1] += y_offset

        # ===== Shoulder Left =====
        ratio = ratios[(1, 5)]
        x_offset = (candidate[1][0] - candidate[5][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[5][1]) * (1. - ratio)
        candidate[[5, 6, 7], 0] += x_offset
        candidate[[5, 6, 7], 1] += y_offset
        hands[0, :, 0] += x_offset
        hands[0, :, 1] += y_offset

        # ===== Upper Arm Right =====
        ratio = ratios[(2, 3)]
        x_offset = (candidate[2][0] - candidate[3][0]) * (1. - ratio)
        y_offset = (candidate[2][1] - candidate[3][1]) * (1. - ratio)
        candidate[[3, 4], 0] += x_offset
        candidate[[3, 4], 1] += y_offset
        hands[1, :, 0] += x_offset
        hands[1, :, 1] += y_offset

        # ===== Forearm Right =====
        ratio = ratios[(3, 4)]
        x_offset = (candidate[3][0] - candidate[4][0]) * (1. - ratio)
        y_offset = (candidate[3][1] - candidate[4][1]) * (1. - ratio)
        candidate[4, 0] += x_offset
        candidate[4, 1] += y_offset
        hands[1, :, 0] += x_offset
        hands[1, :, 1] += y_offset

        # ===== Upper Arm Left =====
        ratio = ratios[(5, 6)]
        x_offset = (candidate[5][0] - candidate[6][0]) * (1. - ratio)
        y_offset = (candidate[5][1] - candidate[6][1]) * (1. - ratio)
        candidate[[6, 7], 0] += x_offset
        candidate[[6, 7], 1] += y_offset
        hands[0, :, 0] += x_offset
        hands[0, :, 1] += y_offset

        # ===== Forearm Left =====
        ratio = ratios[(6, 7)]
        x_offset = (candidate[6][0] - candidate[7][0]) * (1. - ratio)
        y_offset = (candidate[6][1] - candidate[7][1]) * (1. - ratio)
        candidate[7, 0] += x_offset
        candidate[7, 1] += y_offset
        hands[0, :, 0] += x_offset
        hands[0, :, 1] += y_offset

        # ===== Head parts =====
        for (p1, p2) in [(0,14),(0,15),(14,16),(15,17)]:
            ratio = ratios[(p1,p2)]
            x_offset = (candidate[p1][0] - candidate[p2][0]) * (1. - ratio)
            y_offset = (candidate[p1][1] - candidate[p2][1]) * (1. - ratio)
            candidate[p2, 0] += x_offset
            candidate[p2, 1] += y_offset

        # ===== Hips (added) =====
        ratio = ratios[(1, 8)]
        x_offset = (candidate[1][0] - candidate[8][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[8][1]) * (1. - ratio)
        candidate[8, 0] += x_offset
        candidate[8, 1] += y_offset

        ratio = ratios[(1, 11)]
        x_offset = (candidate[1][0] - candidate[11][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[11][1]) * (1. - ratio)
        candidate[11, 0] += x_offset
        candidate[11, 1] += y_offset

        # ===== Legs =====
        ratio = ratios[(8, 9)]
        x_offset = (candidate[9][0] - candidate[8][0]) * (ratio - 1.)
        y_offset = (candidate[9][1] - candidate[8][1]) * (ratio - 1.)
        candidate[[9, 10], 0] += x_offset
        candidate[[9, 10], 1] += y_offset

        ratio = ratios[(9, 10)]
        x_offset = (candidate[10][0] - candidate[9][0]) * (ratio - 1.)
        y_offset = (candidate[10][1] - candidate[9][1]) * (ratio - 1.)
        candidate[10, 0] += x_offset
        candidate[10, 1] += y_offset

        ratio = ratios[(11, 12)]
        x_offset = (candidate[12][0] - candidate[11][0]) * (ratio - 1.)
        y_offset = (candidate[12][1] - candidate[11][1]) * (ratio - 1.)
        candidate[[12, 13], 0] += x_offset
        candidate[[12, 13], 1] += y_offset

        ratio = ratios[(12, 13)]
        x_offset = (candidate[13][0] - candidate[12][0]) * (ratio - 1.)
        y_offset = (candidate[13][1] - candidate[12][1]) * (ratio - 1.)
        candidate[13, 0] += x_offset
        candidate[13, 1] += y_offset

        # rough align
        if has_ref_face and 'faces' in pose and pose['faces'] is not None and pose['faces'].size > 0:
            try:
                src68, to_orig = _to_68x2(pose['faces'])
                scaled68 = _face_scale_only(src68, ref68, candidate[0], alpha=1.0)
                pose['faces'] = to_orig(scaled68)
            except Exception as e:
                print("Reference face conversion failed:", e)
                continue

    return detected_poses
