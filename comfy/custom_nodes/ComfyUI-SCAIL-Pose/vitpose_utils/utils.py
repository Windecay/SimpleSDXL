

import numpy as np
import cv2

# source
# https://github.com/Wan-Video/Wan2.2/blob/e9783574ef77be11fcab9aa5607905402538c08d/wan/modules/animate/preprocess/pose2d_utils.py#L1034

def bbox_from_detector(bbox, input_resolution=(224, 224), rescale=1.25):
    """
    Get center and scale of bounding box from bounding box.
    The expected format is [min_x, min_y, max_x, max_y].
    """
    CROP_IMG_HEIGHT, CROP_IMG_WIDTH = input_resolution
    CROP_ASPECT_RATIO = CROP_IMG_HEIGHT / float(CROP_IMG_WIDTH)

    # center
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    center = np.array([center_x, center_y])

    # scale
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = max(bbox_w * CROP_ASPECT_RATIO, bbox_h)

    scale = np.array([bbox_size / CROP_ASPECT_RATIO, bbox_size]) / 200.0
    # scale = bbox_size / 200.0
    # adjust bounding box tightness
    scale *= rescale
    return center, scale

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # res: (height, width), (rows, cols)
    crop_aspect_ratio = res[0] / float(res[1])
    h = 200 * scale
    w = h / crop_aspect_ratio
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / w
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / w + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return np.array([round(new_pt[0]), round(new_pt[1])], dtype=int) + 1

def crop(img, center, scale, res):
    """
    Crop image according to the supplied bounding box.
    res: [rows, cols]
    """
    # Upper left point
    ul = np.array(transform([1, 1], center, max(scale), res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[1] + 1, res[0] + 1], center, max(scale), res, invert=1)) - 1

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    except Exception as e:
        print(e)

    new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)
    return new_img, new_shape, (old_x, old_y), (new_x, new_y)  # , ul, br


def split_kp2ds_for_aa(kp2ds, ret_face=False):
    kp2ds_body = (kp2ds[[0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]] + kp2ds[[0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]]) / 2
    kp2ds_lhand = kp2ds[91:112]
    kp2ds_rhand = kp2ds[112:133]
    kp2ds_face = kp2ds[22:91]
    if ret_face:
        return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy(), kp2ds_face.copy()
    return kp2ds_body.copy(), kp2ds_lhand.copy(), kp2ds_rhand.copy()


def load_pose_metas_from_kp2ds_seq(kp2ds_seq, width, height):
    metas = []
    last_kp2ds_body = None
    for kps in kp2ds_seq:
        kps = kps.copy()
        kps[:, 0] /= width
        kps[:, 1] /= height
        kp2ds_body, kp2ds_lhand, kp2ds_rhand, kp2ds_face = split_kp2ds_for_aa(kps, ret_face=True)

        # Exclude cases where all values are less than 0
        if last_kp2ds_body is not None and kp2ds_body[:, :2].min(axis=1).max() < 0:
            kp2ds_body = last_kp2ds_body
        last_kp2ds_body = kp2ds_body

        meta = {
            "width": width,
            "height": height,
            "keypoints_body": kp2ds_body,
            "keypoints_left_hand": kp2ds_lhand,
            "keypoints_right_hand": kp2ds_rhand,
            "keypoints_face": kp2ds_face,
        }
        metas.append(meta)
    return metas

def aaposemeta_to_dwpose_scail(meta):
    """
    Convert AA pose metadata to DWpose format matching DWposeDetector output.

    DWpose format:
    - bodies: dict with 'candidate' (n, 24, 2) and 'subset' (n, 24) where subset contains indices
    - hands: array (2*n, 21, 2) - stacked right/left hands
    - faces: array (n, 68, 2)
    """
    # Body keypoints (excluding last 2)
    candidate_body = meta['keypoints_body'][:-2][:, :2]  # (24, 2)
    score_body = meta['keypoints_body'][:-2][:, 2]       # (24,)

    # Create subset: contains joint index if visible, -1 if not
    subset_body = np.arange(len(candidate_body), dtype=float)
    subset_body[score_body <= 0.3] = -1  # Match DWpose threshold

    # Bodies dict with single person (expand to match multi-person format)
    bodies = {
        "candidate": np.expand_dims(candidate_body, axis=0),  # (1, 24, 2)
        "subset": np.expand_dims(subset_body, axis=0)         # (1, 24)
    }

    # Hands: stack right then left (2, 21, 2)
    hands_coords = np.stack([
        meta['keypoints_right_hand'][:, :2],
        meta['keypoints_left_hand'][:, :2]
    ], axis=0)

    hands_score = np.stack([
        meta['keypoints_right_hand'][:, 2],
        meta['keypoints_left_hand'][:, 2]
    ], axis=0)

    # Faces: (1, 68, 2) - skip first face keypoint like DWpose does (24:92 = 68 points)
    faces_coords = np.expand_dims(meta['keypoints_face'][1:][:, :2], axis=0)
    faces_score = np.expand_dims(meta['keypoints_face'][1:][:, 2], axis=0)

    # Match DWpose output structure
    dwpose_format = {
        "bodies": bodies,
        "hands": hands_coords,
        "faces": faces_coords
    }

    # Optional: include scores separately like DWpose does
    score_dict = {
        "body_score": np.expand_dims(score_body, axis=0),
        "hand_score": hands_score,
        "face_score": faces_score
    }

    # Merge score dict into dwpose_format
    dwpose_format.update(score_dict)

    return dwpose_format
