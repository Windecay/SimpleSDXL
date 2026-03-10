from typing import List, Tuple

import cv2
import numpy as np
import torch


def preprocess(img: np.ndarray, out_bbox, input_size: Tuple[int, int] = (192, 256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])

        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        resized_img, scale = top_down_affine(input_size, scale, center, img)

        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std

        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)

    return out_img, out_center, out_scale


def inference(model, img, bs=5):
    all_out = []
    orig_img_count = len(img)
    for _ in range(bs - (orig_img_count % bs)):
        img.append(np.zeros_like(img[0]))
    input = np.stack(img, axis=0).transpose(0, 3, 1, 2)
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    input = torch.from_numpy(input).to(device, dtype)

    out1, out2 = [], []
    for i in range(input.shape[0] // bs):
        curr_batch_output = model(input[i * bs:(i + 1) * bs])
        out1.append(curr_batch_output[0].float())
        out2.append(curr_batch_output[1].float())
    out1, out2 = torch.cat(out1, dim=0)[:orig_img_count], torch.cat(out2, dim=0)[:orig_img_count]
    out1, out2 = out1.float().cpu().detach().numpy(), out2.float().cpu().detach().numpy()
    all_outputs = out1, out2

    for batch_idx in range(len(all_outputs[0])):
        outputs = [all_outputs[i][batch_idx:batch_idx + 1, ...] for i in range(len(all_outputs))]
        all_out.append(outputs)
    return all_out


def postprocess(outputs: List[np.ndarray], model_input_size: Tuple[int, int], center: Tuple[int, int], scale: Tuple[int, int], simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    all_key = []
    all_score = []
    for i in range(len(outputs)):
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])

    return np.array(all_key), np.array(all_score)


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h])
    )
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, output_size: Tuple[int, int], shift: Tuple[float, float] = (0., 0.), inv: bool = False) -> np.ndarray:
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w, h = input_size
    warp_size = (int(w), int(h))

    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores


def inference_pose(model, out_bbox, oriImg, model_input_size=(288, 384)):
    resized_img, center, scale = preprocess(oriImg, out_bbox, model_input_size)
    outputs = inference(model, resized_img)

    keypoints, scores = postprocess(outputs, model_input_size, center, scale)

    return keypoints, scores
