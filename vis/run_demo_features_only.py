"""
只提取和保存特征图的demo版本（不需要OpenGL渲染）
"""
from typing import Dict
import hydra
import cv2
import os
import torch
from tqdm import tqdm
import numpy as np
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from datasets.constants import DETECTRON_CFG
from models.cam_model.fl_net import FLNet
from util.constants import CAM_MODEL_CKPT, DETECTRON_CKPT, IMAGE_MEAN, IMAGE_STD
from vis.dataset import Dataset
from util.data import recursive_to
from util.utils_detectron2 import DefaultPredictor_Lazy
from vis.skelcf_render import SKELCFRender
from sklearn.decomposition import PCA


def fix_prefix_state_dict(st: Dict, model=None):
    new_st = {}
    for k, v in st.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '', 1)
            new_st[new_k] = v
        else:
            new_st[k] = v
    
    # 删除不需要的keys
    keys_to_remove = ['updates', 'decoder.ra_init']
    for k in keys_to_remove:
        if k in new_st:
            del new_st[k]
    
    # 如果提供了模型，根据模型的期望形状来调整axis_flip参数
    if model is not None:
        model_state = model.state_dict()
        for k in list(new_st.keys()):
            if 'axis_flip' in k and k in model_state:
                checkpoint_shape = new_st[k].shape
                model_shape = model_state[k].shape
                
                # 形状不匹配时进行转换
                if checkpoint_shape != model_shape:
                    # [N, 1] -> [N]
                    if len(checkpoint_shape) == 2 and checkpoint_shape[1] == 1 and len(model_shape) == 1:
                        new_st[k] = new_st[k].squeeze(-1)
                    # [N] -> [N, 1]
                    elif len(checkpoint_shape) == 1 and len(model_shape) == 2 and model_shape[1] == 1:
                        new_st[k] = new_st[k].unsqueeze(-1)
    
    return new_st


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img


def _infer_patch_grid(n_patches: int, target_ratio: float = 4 / 3):
    # 根据 patch 数量和目标长宽比，推断 (H, W) 网格
    best_h, best_w = 1, n_patches
    best_err = float('inf')
    for h in range(1, int(np.sqrt(n_patches)) + 1):
        if n_patches % h != 0:
            continue
        w = n_patches // h
        for hh, ww in [(h, w), (w, h)]:
            err = abs((hh / ww) - target_ratio)
            if err < best_err:
                best_h, best_w, best_err = hh, ww, err
    return best_h, best_w


def visualize_vit_layer_features(layer_features, output_dir, img_name, img_ext='.jpg', target_ratio: float = 4 / 3):
    """
    可视化 ViT 各层的特征图并保存
    Args:
        layer_features: List of (B, N, dim) 每层的特征
        output_dir: 输出目录
        img_name: 图像名称（不含扩展名）
        img_ext: 图像扩展名
    """
    # 创建子文件夹
    layer_dir = os.path.join(output_dir, f'{img_name}_vit_layers')
    os.makedirs(layer_dir, exist_ok=True)
    
    num_layers = len(layer_features)
    
    # 保存每一层的特征图
    for layer_idx, features in enumerate(layer_features):
        # 取第一个 batch
        feat = features[0].detach().cpu().numpy()  # (N, dim)
        
        # 使用 PCA 降维到 3 维用于 RGB 可视化
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat)  # (N, 3)
        
        # 归一化到 0-255
        feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
        feat_pca = (feat_pca * 255).astype(np.uint8)
        
        # 推断网格大小
        n_patches = feat.shape[0]
        h_patches, w_patches = _infer_patch_grid(n_patches, target_ratio=target_ratio)
        
        # 重塑为图像
        feat_img = feat_pca.reshape(h_patches, w_patches, 3)
        
        # 调整大小并保存
        feat_img = cv2.resize(feat_img, (192, 256), interpolation=cv2.INTER_LINEAR)
        layer_fname = os.path.join(layer_dir, f'layer_{layer_idx:02d}{img_ext}')
        cv2.imwrite(layer_fname, feat_img)
    
    # 创建汇总图
    summary_img = create_vit_layers_summary(layer_features, num_cols=8, target_ratio=target_ratio)
    summary_fname = os.path.join(output_dir, f'{img_name}_01b_vit_layers_summary{img_ext}')
    cv2.imwrite(summary_fname, summary_img)
    
    return layer_dir, summary_fname


def create_vit_layers_summary(layer_features, num_cols=8, target_ratio: float = 4 / 3):
    """
    创建 ViT 各层特征的汇总图
    """
    num_layers = len(layer_features)
    num_rows = (num_layers + num_cols - 1) // num_cols
    
    # 每个小图的大小
    cell_h = 64
    cell_w = max(1, int(round(cell_h / target_ratio)))
    
    # 创建画布
    canvas = np.ones((num_rows * cell_h, num_cols * cell_w, 3), dtype=np.uint8) * 255
    
    for idx, features in enumerate(layer_features):
        row = idx // num_cols
        col = idx % num_cols
        
        # 取第一个 batch
        feat = features[0].detach().cpu().numpy()  # (N, dim)
        
        # PCA 降维
        pca = PCA(n_components=3)
        feat_pca = pca.fit_transform(feat)
        feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
        feat_pca = (feat_pca * 255).astype(np.uint8)
        
        # 推断网格
        n_patches = feat.shape[0]
        h_patches, w_patches = _infer_patch_grid(n_patches, target_ratio=target_ratio)
        
        # 重塑并调整大小
        feat_img = feat_pca.reshape(h_patches, w_patches, 3)
        feat_img = cv2.resize(feat_img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        
        # 放置到画布上
        y1 = row * cell_h
        x1 = col * cell_w
        canvas[y1:y1+cell_h, x1:x1+cell_w] = feat_img
        
        # 添加层号标签
        cv2.putText(canvas, f'L{idx}', (x1+2, y1+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    return canvas


def visualize_decoder_attention(decoder_attentions, output_dir, img_name, img_ext='.jpg', bg_img=None, bbox_center=None, bbox_size=None, img_size=None):
    """
    可视化 Decoder Cross-Attention 权重（显示decoder关注的空间区域）
    """
    if decoder_attentions is None or len(decoder_attentions) == 0:
        return None
    
    num_layers = len(decoder_attentions)
    token_names = ['Pose', 'Beta', 'Cam']
    
    # 推断patch网格
    first_attn = decoder_attentions[0]
    if first_attn is None:
        return None
    
    n_patches = first_attn.shape[-1]
    h_patches, w_patches = _infer_patch_grid(n_patches, target_ratio=4 / 3)
    
    # 为每层的每个token创建attention map
    for layer_idx, attn in enumerate(decoder_attentions):
        if attn is None:
            continue
            
        attn_cpu = attn[0].cpu().numpy()  # (n_heads, n_tokens, n_patches)
        attn_avg = attn_cpu.mean(axis=0)  # (n_tokens, n_patches)
        
        for token_idx in range(min(attn_avg.shape[0], len(token_names))):
            token_attn = attn_avg[token_idx]
            token_attn = (token_attn - token_attn.min()) / (token_attn.max() - token_attn.min() + 1e-8)
            attn_map = token_attn.reshape(h_patches, w_patches)
            attn_map_vis = cv2.resize(attn_map, (192, 256), interpolation=cv2.INTER_LINEAR)
            attn_heatmap = (attn_map_vis * 255).astype(np.uint8)
            attn_heatmap = cv2.applyColorMap(attn_heatmap, cv2.COLORMAP_JET)
            
            if bg_img is not None and bbox_center is not None and bbox_size is not None:
                output_img = bg_img.copy()
                cx, cy = int(bbox_center[0]), int(bbox_center[1])
                bbox_h = int(bbox_size)
                bbox_w = int(bbox_h * (w_patches / h_patches))
                
                x1 = max(0, cx - bbox_w // 2)
                y1 = max(0, cy - bbox_h // 2)
                x2 = min(bg_img.shape[1], cx + bbox_w // 2)
                y2 = min(bg_img.shape[0], cy + bbox_h // 2)
                
                if y2 > y1 and x2 > x1:
                    attn_resized = cv2.resize(attn_heatmap, (x2 - x1, y2 - y1))
                    output_img[y1:y2, x1:x2] = cv2.addWeighted(
                        output_img[y1:y2, x1:x2], 0.5, attn_resized, 0.5, 0
                    )
                    label = f'Layer{layer_idx} - {token_names[token_idx]} Token'
                    cv2.putText(output_img, label, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    attn_fname = os.path.join(output_dir, 
                        f'{img_name}_L{layer_idx}_{token_names[token_idx]}_attention{img_ext}')
                    cv2.imwrite(attn_fname, output_img)
    
    # 创建汇总图
    summary_rows = []
    for layer_idx, attn in enumerate(decoder_attentions):
        if attn is None:
            continue
        attn_cpu = attn[0].cpu().numpy()
        attn_avg = attn_cpu.mean(axis=0)
        pose_attn = attn_avg[0]
        pose_attn = (pose_attn - pose_attn.min()) / (pose_attn.max() - pose_attn.min() + 1e-8)
        attn_map = pose_attn.reshape(h_patches, w_patches)
        attn_map_vis = cv2.resize(attn_map, (96, 128), interpolation=cv2.INTER_LINEAR)
        attn_heatmap = (attn_map_vis * 255).astype(np.uint8)
        attn_heatmap = cv2.applyColorMap(attn_heatmap, cv2.COLORMAP_JET)
        cv2.putText(attn_heatmap, f'L{layer_idx}', (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        summary_rows.append(attn_heatmap)
    
    if summary_rows:
        summary_img = np.hstack(summary_rows)
        summary_fname = os.path.join(output_dir, f'{img_name}_decoder_attention_summary{img_ext}')
        cv2.imwrite(summary_fname, summary_img)
        return summary_fname
    return None


def visualize_decoder_features(decoder_features, output_dir, img_name, img_ext='.jpg'):
    """
    可视化 Decoder Transformer 各层的token特征并保存
    Args:
        decoder_features: List of (B, N_tokens, dim) 每层的token特征 (N_tokens通常是3: pose, betas, cam)
        output_dir: 输出目录
        img_name: 图像名称（不含扩展名）
        img_ext: 图像扩展名
    """
    if decoder_features is None or len(decoder_features) == 0:
        return None
    
    num_layers = len(decoder_features)
    num_tokens = decoder_features[0].shape[1]  # 通常是3
    
    # 创建汇总图：每行代表一层，每列代表一个token
    cell_h = 128
    cell_w = 128
    canvas = np.ones((num_layers * cell_h, num_tokens * cell_w, 3), dtype=np.uint8) * 255
    
    base_token_names = ['Pose', 'Beta', 'Cam']
    token_names = [
        base_token_names[i] if i < len(base_token_names) else f'Token{i}'
        for i in range(num_tokens)
    ]
    
    # 使用所有层的token做一次PCA，保证颜色稳定且可比
    all_tokens = []
    for layer_feat in decoder_features:
        feat = layer_feat[0].detach().cpu().numpy()  # (N_tokens, dim)
        all_tokens.append(feat)
    all_tokens = np.concatenate(all_tokens, axis=0)  # (num_layers * num_tokens, dim)
    
    if all_tokens.shape[1] > 3 and all_tokens.shape[0] >= 2:
        pca = PCA(n_components=3)
        all_tokens_pca = pca.fit_transform(all_tokens)
    else:
        all_tokens_pca = all_tokens
        # 如果不足3维，补齐
        while all_tokens_pca.shape[1] < 3:
            all_tokens_pca = np.concatenate([all_tokens_pca, np.zeros((all_tokens_pca.shape[0], 1))], axis=1)
    
    # 全局归一化，保持跨层对比一致
    all_tokens_pca = (all_tokens_pca - all_tokens_pca.min()) / (all_tokens_pca.max() - all_tokens_pca.min() + 1e-8)
    
    for layer_idx in range(num_layers):
        for token_idx in range(num_tokens):
            idx = layer_idx * num_tokens + token_idx
            color_value = (all_tokens_pca[idx] * 255).astype(np.uint8)
            
            # 创建单色块
            cell = np.ones((cell_h, cell_w, 3), dtype=np.uint8)
            cell[:, :] = color_value
            
            # 放置到画布上
            y1 = layer_idx * cell_h
            x1 = token_idx * cell_w
            canvas[y1:y1+cell_h, x1:x1+cell_w] = cell
            
            # 添加标签
            label = f'L{layer_idx}-{token_names[token_idx]}'
            cv2.putText(canvas, label, (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 保存汇总图
    summary_fname = os.path.join(output_dir, f'{img_name}_decoder_features{img_ext}')
    cv2.imwrite(summary_fname, canvas)
    
    return summary_fname


def visualize_vit_features(vit_features, img_size, save_path=None, bg_img=None, bbox_center=None, bbox_size=None):
    """
    可视化 ViT 提取的特征图 - 使用多种方法
    Args:
        vit_features: (B, N, dim) ViT patch 特征
        img_size: (H, W) 原图尺寸
        save_path: 保存路径 (可选，为 None 时只返回图像不保存)
        bg_img: 背景原图 (可选，用于叠加显示)
        bbox_center: bbox 中心坐标 (可选)
        bbox_size: bbox 大小 (可选)
    Returns:
        feature_img: 可视化后的特征图 (H, W, 3)
    """
    # 取第一个 batch
    features = vit_features[0].detach().cpu().numpy()  # (N, dim)
    n_patches = features.shape[0]

    # 推断 patch 网格 (默认按 256x192 → 4:3 比例)
    h_patches, w_patches = _infer_patch_grid(n_patches, target_ratio=4 / 3)

    # === 方法1: 特征L2范数热力图（更能突出人形区域）===
    feature_norm = np.linalg.norm(features, axis=1)  # (N,)
    feature_norm = (feature_norm - feature_norm.min()) / (feature_norm.max() - feature_norm.min() + 1e-8)
    heatmap = feature_norm.reshape(h_patches, w_patches)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 应用JET颜色映射
    
    # === 方法2: PCA降维到3维，每个通道单独归一化（更强对比度）===
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features)  # (N, 3)
    
    # 每个通道单独归一化（更好的对比度）
    for i in range(3):
        channel = features_pca[:, i]
        features_pca[:, i] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
    features_pca = (features_pca * 255).astype(np.uint8)
    pca_img = features_pca.reshape(h_patches, w_patches, 3)
    
    # === 合并两种可视化：左边热力图，右边PCA ===
    heatmap_resized = cv2.resize(heatmap_color, (w_patches * 8, h_patches * 8), interpolation=cv2.INTER_LINEAR)
    pca_resized = cv2.resize(pca_img, (w_patches * 8, h_patches * 8), interpolation=cv2.INTER_LINEAR)
    combined_vis = np.hstack([heatmap_resized, pca_resized])

    # 如果提供了背景图和 bbox 信息，将热力图叠加到原图的 bbox 位置
    if bg_img is not None and bbox_center is not None and bbox_size is not None:
        output_img = bg_img.copy()

        # 计算 bbox 范围
        cx, cy = int(bbox_center[0]), int(bbox_center[1])
        bbox_h = int(bbox_size)
        bbox_w = max(1, int(bbox_h * (w_patches / h_patches)))

        x1 = max(0, cx - bbox_w // 2)
        y1 = max(0, cy - bbox_h // 2)
        x2 = min(bg_img.shape[1], cx + bbox_w // 2)
        y2 = min(bg_img.shape[0], cy + bbox_h // 2)

        if y2 > y1 and x2 > x1:
            # 使用热力图叠加（更能突出人形）
            heatmap_overlay = cv2.resize(heatmap_color, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            output_img[y1:y2, x1:x2] = cv2.addWeighted(
                output_img[y1:y2, x1:x2], 0.4, heatmap_overlay, 0.6, 0
            )
        
        # 在右侧添加独立的特征可视化图
        vis_h = min(200, bg_img.shape[0] // 3)
        vis_w = vis_h * 2  # 热力图+PCA并排
        combined_small = cv2.resize(combined_vis, (vis_w, vis_h // 2 * 2))
        
        # 放置在图像右下角
        y_start = bg_img.shape[0] - combined_small.shape[0] - 10
        x_start = bg_img.shape[1] - combined_small.shape[1] - 10
        
        if y_start > 0 and x_start > 0:
            output_img[y_start:y_start+combined_small.shape[0], 
                      x_start:x_start+combined_small.shape[1]] = combined_small

        if save_path is not None:
            cv2.imwrite(save_path, output_img)
        return output_img

    # 调整到指定大小
    feature_img = cv2.resize(heatmap_color, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    if save_path is not None:
        cv2.imwrite(save_path, feature_img)
    return feature_img


class FeatureExtractor:
    def __init__(self, cfg, threshold=0.25):
        self.cfg = cfg 
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"Using device: {self.device}")
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT, map_location='cpu')['state_dict']
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def init_model(self):
        model = SKELCFRender(self.cfg)
        checkpoint = torch.load(self.cfg.trainer.ckpt_path, map_location='cpu')['ema_model']
        st = fix_prefix_state_dict(checkpoint, model=model)
        model.load_state_dict(st, strict=True)
        model = model.to(self.device)
        model.eval()
        return model
    
    def init_detector(self, threshold):
        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)
        return cam_int

    def process_image(self, img_path, output_img_folder):
        img_cv2 = cv2.imread(str(img_path))
        fname, img_ext = os.path.splitext(os.path.basename(img_path))

        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.7)
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # image h & w
        img_h, img_w = img_cv2.shape[:2]

        # bbox center 
        center_x = bbox_center[:, 0]
        center_y = bbox_center[:, 1]

        center_mask = (
            (center_x > 0.1 * img_w) & (center_x < 0.9 * img_w) &
            (center_y > 0.1 * img_h) & (center_y < 0.9 * img_h)
        )
        
        if center_mask.sum() > 0 :
            areas = areas[center_mask]
            boxes = boxes[center_mask]
            bbox_scale = bbox_scale[center_mask]
            bbox_center = bbox_center[center_mask]
        
        num_keep = min(self.cfg.misc.num_keep, len(areas))
        keep = areas.argsort()[-num_keep:][::-1]
        boxes = boxes[keep]
        bbox_scale = bbox_scale[keep]
        bbox_center = bbox_center[keep]
        
        # Get Camera intrinsics
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(cfg=self.cfg, img_cv2=img_cv2, bbox_center=bbox_center, 
                            bbox_scale=bbox_scale, cam_int=cam_int, train=False, img_path=img_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            
            with torch.no_grad():
                enc_params, dec_params, per_layer_params, vit_features, vit_layers, decoder_features, decoder_attentions = self.model(batch, return_vit_layers=True)

            print(f"✓ Processed {fname}")
            
            # ========== 只保存特征图可视化 ==========
            
            # 1. 保存 ViT 特征可视化
            if vit_features is not None:
                vit_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_01_vit_features{img_ext}')
                bbox_center_0 = batch['box_center'][0].cpu().numpy()
                bbox_size_0 = batch['bbox_size'][0].cpu().item()
                visualize_vit_features(
                    vit_features, 
                    (int(img_h), int(img_w)), 
                    vit_fname,
                    bg_img=img_cv2,
                    bbox_center=bbox_center_0,
                    bbox_size=bbox_size_0
                )
                print(f"  → ViT features saved")
            
            # 2. 保存 ViT 各层特征可视化
            if vit_layers is not None and len(vit_layers) > 0:
                visualize_vit_layer_features(
                    vit_layers,
                    output_img_folder,
                    os.path.basename(fname),
                    img_ext
                )
                print(f"  → ViT layer features saved ({len(vit_layers)} layers)")
            
            # 3. 保存 Decoder Cross-Attention 可视化（显示关注的空间区域）
            if decoder_attentions is not None and len(decoder_attentions) > 0:
                bbox_center_0 = batch['box_center'][0].cpu().numpy()
                bbox_size_0 = batch['bbox_size'][0].cpu().item()
                visualize_decoder_attention(
                    decoder_attentions,
                    output_img_folder,
                    os.path.basename(fname),
                    img_ext,
                    bg_img=img_cv2,
                    bbox_center=bbox_center_0,
                    bbox_size=bbox_size_0,
                    img_size=(int(img_h), int(img_w))
                )
                print(f"  → Decoder attention maps saved ({len(decoder_attentions)} layers)")
            
            # 4. 保存 Decoder Token 特征（色块版本，作为补充）
            if decoder_features is not None and len(decoder_features) > 0:
                visualize_decoder_features(
                    decoder_features,
                    output_img_folder,
                    os.path.basename(fname),
                    img_ext
                )
                print(f"  → Decoder token features saved ({len(decoder_features)} layers)")

    def run_on_images(self, image_folder, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        
        print(f"\nFound {len(images_list)} images to process")
        print(f"Output folder: {out_folder}\n")
        
        for ind, img_path in tqdm(enumerate(images_list), desc="Extracting features", total=len(images_list)):
            self.process_image(img_path, out_folder)


@hydra.main(version_base='1.2', config_path="../config", config_name="vis.yaml")
def main(cfg):
    extractor = FeatureExtractor(cfg=cfg)
    extractor.run_on_images(cfg.misc.image_folder, cfg.misc.output_folder)
    print("\n✓ All features extracted successfully!")
    
if __name__=='__main__':
    main()

