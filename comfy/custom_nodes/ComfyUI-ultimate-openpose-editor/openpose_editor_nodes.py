import json
import torch
import numpy as np
from .util import draw_pose_json, draw_pose, extend_scalelist, pose_normalized

OpenposeJSON = dict

class OpenposeEditorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "resolution_x": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 12800,
                    "tooltip": "Resolution X. -1 means use the original resolution."
                }),
                "pose_marker_size": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 100
                }),
                "face_marker_size": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 100
                }),
                "hand_marker_size": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100
                }),
                "hands_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "body_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "head_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "overall_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "scalelist_behavior": (["poses", "images"], {"default": "poses", "tooltip": "When the scale input is a list, this determines how the scale list takes effect, the differences appear when there are multiple persons(poses) in one image."}),
                "match_scalelist_method": (["no extend", "loop extend", "clamp extend"], {"default": "loop extend", "tooltip": "Match the scale list to the input poses or images when the scale list length is shorter. No extend: Beyound the scale list will be 1.0. Loop: Loop the scale list to match the poses or images length. Clamp: Use the last scale value to extend the scale list."}),
                "only_scale_pose_index": ("INT", {
                    "default": 99,
                    "min": -100,
                    "max": 100,
                    "tooltip": "For multiple poses in one image, the scale will be only applied at desired index. If set to a number larger than the number of poses in the image, the scale will be applied to all poses. Negative number will apply to the pose from the end."
                }),
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
            },
        }

    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    OUTPUT_NODE = True
    FUNCTION = "load_pose"
    CATEGORY = "ultimate-openpose"

    def load_pose(self, show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size, hand_marker_size, hands_scale, body_scale, head_scale, overall_scale, scalelist_behavior, match_scalelist_method, only_scale_pose_index, POSE_JSON: str, POSE_KEYPOINT=None) -> tuple[OpenposeJSON]:
        '''
        priority output is: POSE_JSON > POSE_KEYPOINT
        priority edit is: POSE_KEYPOINT > POSE_JSON
        '''
        if POSE_JSON:
            POSE_JSON = POSE_JSON.replace("'",'"').replace('None','[]')
            POSE_PASS = POSE_JSON
            if POSE_KEYPOINT is not None:
                POSE_PASS = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')

            # parse the JSON
            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, POSE_PASS, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(POSE_PASS)
            pose_imgs, POSE_PASS_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(POSE_PASS_SCALED, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), POSE_PASS_SCALED, json.dumps(POSE_PASS_SCALED,indent=4))
                }
        elif POSE_KEYPOINT is not None:
            POSE_JSON = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')
            hands_scalelist, body_scalelist, head_scalelist, overall_scalelist = extend_scalelist(
                scalelist_behavior, POSE_JSON, hands_scale, body_scale, head_scale, overall_scale,
                match_scalelist_method, only_scale_pose_index)
            normalized_pose_json = pose_normalized(POSE_JSON)
            pose_imgs, POSE_SCALED = draw_pose_json(normalized_pose_json, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scalelist, body_scalelist, head_scalelist, overall_scalelist)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [json.dumps(POSE_SCALED, indent=4)]},
                    "result": (torch.from_numpy(pose_imgs_np), POSE_SCALED, json.dumps(POSE_SCALED, indent=4))
                }

        # otherwise output blank images
        W=512
        H=768
        pose_draw = dict(bodies={'candidate':[], 'subset':[]}, faces=[], hands=[])
        pose_out = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
        people=[dict(people=[pose_out], canvas_height=H, canvas_width=W)]

        W_scaled = resolution_x
        if resolution_x < 64:
            W_scaled = W
        H_scaled = int(H*(W_scaled*1.0/W))
        pose_img = [draw_pose(pose_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
        pose_img_np = np.array(pose_img).astype(np.float32) / 255

        return {
                "ui": {"POSE_JSON": people},
                "result": (torch.from_numpy(pose_img_np), people, json.dumps(people))
        }
