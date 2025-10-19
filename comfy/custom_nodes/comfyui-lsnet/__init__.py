import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import requests
from tqdm import tqdm
import urllib.request
sys.path.append(os.path.dirname(__file__))

import folder_paths

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model

from lsnet_model import lsnet_artist  # noqa: F401

from inference_artist import (
    load_checkpoint_state,
    normalize_state_dict_keys,
    resolve_num_classes,
    resolve_feature_dim,
    load_class_mapping
)

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class LSNetModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        base_dir = os.path.join(folder_paths.models_dir, 'lsnet')
        subfolders = []
        if os.path.exists(base_dir):
            subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        
        return {
            "required": {
                "model_folder": (subfolders, {"default": subfolders[0] if subfolders else ""}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("LSNET_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LSNet"

    def load(self, model_folder, device):
        base_dir = os.path.join(folder_paths.models_dir, 'lsnet')
        model_dir = os.path.join(base_dir, model_folder)
        checkpoint_path = os.path.join(model_dir, "best_checkpoint.pth")
        csv_path = os.path.join(model_dir, "class_mapping.csv")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")
        class_mapping = load_class_mapping(csv_path)
        state_dict = load_checkpoint_state(checkpoint_path)
        state_dict = normalize_state_dict_keys(state_dict)
        num_classes = resolve_num_classes(None, class_mapping, state_dict)
        feature_dim = resolve_feature_dim(None, state_dict)
        model = create_model(
            'lsnet_xl_artist',
            pretrained=False,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        config = resolve_data_config({}, model=model)
        transform = create_transform(**config)
        model_bundle = {
            'model': model,
            'transform': transform,
            'class_mapping': class_mapping,
            'device': device
        }

        return (model_bundle,)

class LSNetModelDownloader:
    @classmethod
    def INPUT_TYPES(s):
        file_types = ["best_checkpoint.pth"]  # 只显示一个文件选项
        return {
            "required": {
                "file_to_download": (file_types, {"default": "best_checkpoint.pth"}),
                "device": ("STRING", {"default": "cuda"}),
                "force_download": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LSNET_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "download_and_load"
    CATEGORY = "LSNet"

    FILE_URLS = {
        "best_checkpoint.pth": "https://www.modelscope.cn/models/Heathcliff02/Kaloscope/resolve/master/best_checkpoint.pth",
        "class_mapping.csv": "https://www.modelscope.cn/models/Heathcliff02/Kaloscope/resolve/master/class_mapping.csv"
    }

    def download_file(self, url, destination):
        """从URL下载文件到目标位置"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(destination), exist_ok=True)

            print(f"开始下载文件: {os.path.basename(destination)} 从 {url}")

            response = urllib.request.urlopen(url)
            file_size = int(response.getheader('Content-Length', 0))

            with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                with open(destination, 'wb') as file:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        file.write(chunk)
                        pbar.update(len(chunk))

            print(f"文件已下载完成: {destination}")
            return True
        except Exception as e:
            print(f"下载失败: {str(e)}")
            if os.path.exists(destination):
                os.remove(destination)
            return False

    def download_and_load(self, file_to_download, device, force_download):
        # 设置下载路径 - 默认使用kaloscope文件夹
        base_dir = os.path.join(folder_paths.models_dir, 'lsnet', 'kaloscope')
        checkpoint_path = os.path.join(base_dir, "best_checkpoint.pth")
        csv_path = os.path.join(base_dir, "class_mapping.csv")

        both_files_exist = os.path.exists(checkpoint_path) and os.path.exists(csv_path)

        if file_to_download == "best_checkpoint.pth" and (force_download or not both_files_exist):
            files_to_download = ["best_checkpoint.pth", "class_mapping.csv"]
            for file_name in files_to_download:
                file_path = checkpoint_path if file_name == "best_checkpoint.pth" else csv_path
                if force_download or not os.path.exists(file_path):
                    print(f"正在下载 {file_name} 到: {file_path}")
                    if not self.download_file(self.FILE_URLS[file_name], file_path):
                        raise RuntimeError(f"无法下载文件: {file_path}")
                else:
                    print(f"文件 {file_name} 已存在，跳过下载")

        files_exist = True
        missing_files = []

        if not os.path.exists(checkpoint_path):
            files_exist = False
            missing_files.append(checkpoint_path)

        if not os.path.exists(csv_path):
            files_exist = False
            missing_files.append(csv_path)

        if not files_exist:
            missing_info = "\n".join(missing_files)
            raise FileNotFoundError(f"加载模型需要的以下文件未找到:\n{missing_info}\n请先下载所有必要文件")

        try:
            class_mapping = load_class_mapping(csv_path)
            state_dict = load_checkpoint_state(checkpoint_path)
            state_dict = normalize_state_dict_keys(state_dict)
            num_classes = resolve_num_classes(None, class_mapping, state_dict)
            feature_dim = resolve_feature_dim(None, state_dict)

            model = create_model(
                'lsnet_xl_artist',
                pretrained=False,
                num_classes=num_classes,
                feature_dim=feature_dim,
            )

            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)

            model_bundle = {
                'model': model,
                'transform': transform,
                'class_mapping': class_mapping,
                'device': device
            }

            return (model_bundle,)
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise RuntimeError(f"模型加载失败: {str(e)}")

class LSNetArtistInferenceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("LSNET_MODEL",),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tag_string", "json_output")
    FUNCTION = "process"
    CATEGORY = "LSNet"

    def process(self, image, model, top_k, threshold):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        class_mapping = model_bundle['class_mapping']
        device = model_bundle['device']

        if image.ndim == 4:
            image = image[0]
        image = (image * 255).clamp(0, 255).byte().cpu().numpy()
        pil_image = Image.fromarray(image)

        # Preprocess image
        image_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Classify
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            logits = model(image_tensor, return_features=False)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)

            results = []
            for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
                if prob >= threshold:
                    class_id = int(idx)
                    class_name = class_mapping.get(class_id, f"Class {class_id}")
                    results.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'probability': float(prob)
                    })

            # Limit to top_k if more results after filtering
            if len(results) > top_k:
                results = results[:top_k]

        # Prepare outputs
        tags = [res['class_name'] for res in results]
        tag_string = ",".join(tags)
        tag_dict = {res['class_name']: res['probability'] for res in results}
        json_output = json.dumps(tag_dict, ensure_ascii=False)

        return (tag_string, json_output)

class LSNetArtistSimilarityNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "model": ("LSNET_MODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("similarity_json",)
    FUNCTION = "process"
    CATEGORY = "LSNet"

    def process(self, processed_image, reference_images, model):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        def image_to_tensor(img):
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img)
            return transform(pil_img).unsqueeze(0)

        processed_tensor = image_to_tensor(processed_image)
        with torch.no_grad():
            processed_tensor = processed_tensor.to(device)
            processed_features = model(processed_tensor, return_features=True).cpu().numpy()[0]

        references = []
        similarities = []
        num_refs = reference_images.shape[0] if reference_images.ndim == 4 else 1
        for i in range(num_refs):
            ref_img = reference_images[i] if reference_images.ndim == 4 else reference_images
            ref_tensor = image_to_tensor(ref_img)
            with torch.no_grad():
                ref_tensor = ref_tensor.to(device)
                ref_features = model(ref_tensor, return_features=True).cpu().numpy()[0]
            references.append(ref_features.tolist())
            sim = np.dot(processed_features, ref_features) / (np.linalg.norm(processed_features) * np.linalg.norm(ref_features))
            similarities.append(float(sim))

        result = {
            "processed_features": processed_features.tolist(),
            "reference_features": references,
            "similarities": similarities
        }
        json_output = json.dumps(result, ensure_ascii=False)

        return (json_output,)

class LSNetCommonFeaturesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_images": ("IMAGE",),
                "model": ("LSNET_MODEL",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("common_features",)
    FUNCTION = "process"
    CATEGORY = "LSNet"

    def process(self, reference_images, model):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        def image_to_tensor(img):
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img)
            return transform(pil_img).unsqueeze(0)

        references = []
        num_refs = reference_images.shape[0] if reference_images.ndim == 4 else 1
        for i in range(num_refs):
            ref_img = reference_images[i] if reference_images.ndim == 4 else reference_images
            ref_tensor = image_to_tensor(ref_img)
            with torch.no_grad():
                ref_tensor = ref_tensor.to(device)
                ref_features = model(ref_tensor, return_features=True).cpu().numpy()[0]
            references.append(ref_features)

        if references:
            common_features = np.mean(np.array(references), axis=0)
        else:
            common_features = np.zeros(384)
        return (torch.tensor(common_features),)

class LSNetClusteringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (["kmeans", "dbscan", "hierarchical"], {"default": "kmeans"}),
                "n_clusters": ("INT", {"default": 10, "min": 2, "max": 100}),
                "eps": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0}),
                "min_samples": ("INT", {"default": 5, "min": 1, "max": 50}),
                "visualize": ("BOOLEAN", {"default": True}),
                "viz_method": (["tsne", "pca"], {"default": "tsne"}),
                "perplexity": ("INT", {"default": 30, "min": 5, "max": 100}),
            },
            "optional": {
                "group_1": ("TENSOR",),
                "group_2": ("TENSOR",),
                "group_3": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("clustering_json", "visualization")
    FUNCTION = "cluster"
    CATEGORY = "LSNet"

    def cluster(self, method, n_clusters, eps, min_samples, visualize, viz_method, perplexity, group_1=None, group_2=None, group_3=None):
        groups = []
        group_sizes = []
        for g in [group_1, group_2, group_3]:
            if g is not None:
                groups.append(g.cpu().numpy())
                group_sizes.append(g.shape[0])

        if not groups:
            return (json.dumps({"error": "No groups provided"}), torch.zeros(1, 64, 64, 3))

        features_np = np.vstack(groups)
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(features_np)
            centers = clusterer.cluster_centers_
        elif method == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(features_np)
            centers = None
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(features_np)
            centers = None

        result = {
            "method": method,
            "n_samples": len(features_np),
            "group_sizes": group_sizes,
            "labels": labels.tolist(),
        }
        if centers is not None:
            result["centers"] = centers.tolist()

        json_output = json.dumps(result, ensure_ascii=False)

        if visualize and len(features_np) > 1:
            if viz_method == "tsne":
                reducer = TSNE(n_components=2, perplexity=min(perplexity, len(features_np)-1), random_state=42)
            else:
                reducer = PCA(n_components=2, random_state=42)
            
            reduced_features = reducer.fit_transform(features_np)
            
            plt.figure(figsize=(10, 8))
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                           color=color, label=f'Cluster {label}', alpha=0.7)
            
            plt.title(f'{method.upper()} Clustering ({viz_method.upper()})')
            plt.legend()
            plt.tight_layout()
            
            fig = plt.gcf()
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            pil_image = Image.fromarray(img_array)
            plt.close()
            
            viz_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            if viz_tensor.ndim == 3:
                viz_tensor = viz_tensor.unsqueeze(0)
        else:
            viz_tensor = torch.zeros(1, 64, 64, 3)

        return (json_output, viz_tensor)

class LSNetFeatureComparisonNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("LSNET_MODEL",),
            },
            "optional": {
                "group_1": ("TENSOR",),
                "group_2": ("TENSOR",),
                "group_3": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_json",)
    FUNCTION = "compare"
    CATEGORY = "LSNet"

    def compare(self, image, model, group_1=None, group_2=None, group_3=None):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        if image.ndim == 4:
            image = image[0]
        image_np = (image * 255).clamp(0, 255).byte().cpu().numpy()
        pil_image = Image.fromarray(image_np)
        image_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            query_features = model(image_tensor, return_features=True).cpu().numpy()[0]

        groups = []
        for g in [group_1, group_2, group_3]:
            if g is not None:
                groups.append(g.cpu().numpy())

        if not groups:
            return (json.dumps({"error": "No groups provided"}),)

        similarities = []
        for group_feat in groups:
            sim = np.dot(query_features, group_feat) / (np.linalg.norm(query_features) * np.linalg.norm(group_feat))
            similarities.append(float(sim))

        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]
        result = {
            "best_group_index": int(best_index),
            "best_similarity": best_similarity,
            "all_similarities": similarities
        }
        json_output = json.dumps(result, ensure_ascii=False)

        return (json_output,)

class LSNetArtistImageConnector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stacked_images",)
    FUNCTION = "connect"
    CATEGORY = "LSNet"

    def connect(self, image_1, image_2, image_3):
        def normalize_image(img):
            if img.ndim == 4:
                img = img[0]
            return img.unsqueeze(0)

        img1 = normalize_image(image_1)
        img2 = normalize_image(image_2)
        img3 = normalize_image(image_3)

        stacked = torch.cat([img1, img2, img3], dim=0)
        return (stacked,)

NODE_CLASS_MAPPINGS = {
    "LSNetModelLoader": LSNetModelLoader,
    "LSNetModelDownloader": LSNetModelDownloader,
    "LSNetArtistInference": LSNetArtistInferenceNode,
    "LSNetArtistSimilarity": LSNetArtistSimilarityNode,
    "LSNetCommonFeatures": LSNetCommonFeaturesNode,
    "LSNetClustering": LSNetClusteringNode,
    "LSNetFeatureComparison": LSNetFeatureComparisonNode,
    "LSNetArtistImageConnector": LSNetArtistImageConnector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LSNetModelLoader": "LSNet Model Loader",
    "LSNetModelDownloader": "LSNet Model (Down)loader",
    "LSNetArtistInference": "LSNet Artist Inference",
    "LSNetArtistSimilarity": "LSNet Artist Similarity",
    "LSNetCommonFeatures": "LSNet Common Features",
    "LSNetClustering": "LSNet Clustering",
    "LSNetFeatureComparison": "LSNet Feature Comparison",
    "LSNetArtistImageConnector": "LSNet Image Connector"
}
