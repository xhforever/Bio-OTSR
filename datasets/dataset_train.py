import os
import warnings
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset

from util.misc import resize_image
from util.pylogger import get_pylogger
from .configs import DATASET_FOLDERS, DATASET_FILES
from .constants import NUM_JOINTS, NUM_BETAS, NUM_PARAMS_SMPL
from .utils import expand_to_aspect_ratio, get_example
from torchvision.transforms import Normalize
from lib.body_models.skel.kin_skel import BIO_OTSR_CONFIG

logger = get_pylogger(__name__)

class DatasetTrainAllLabels(Dataset):
    def __init__(self, cfg, ds_name, is_train=True):
        super(DatasetTrainAllLabels, self).__init__()

        self.ds_name = ds_name
        self.is_train = is_train
        self.cfg = cfg
        self.img_size = cfg.policy.img_patch_size # 256
        self.mean = 255. * np.array(cfg.policy.img_mean) 
        self.std = 255. * np.array(cfg.policy.img_std)
        self.normalize_img = Normalize(mean=cfg.policy.img_mean, std=cfg.policy.img_std)
        self.border_mode = cv2.BORDER_CONSTANT
        self.img_dir = DATASET_FOLDERS[ds_name]
        
        self.labels = np.load(DATASET_FILES[self.is_train][self.ds_name], allow_pickle=True)
    
        self.imgname = self.labels['imgname']
        self.scale = self.labels['scale'] # (N,) or (N,2)
        self.center = self.labels['center']  # (N,) or (N,2)

        # 1. load skel poses 
        if 'skel_poses' in self.labels:
            self.skel_poses = self.labels['skel_poses']
        elif 'orig_poses' in self.labels:
            self.skel_poses = self.labels['orig_poses']
        else:
            raise ValueError('Unknown skel poses key')
        
        # 2. load skel betas
        if 'skel_betas' in self.labels:
            self.skel_betas = self.labels['skel_betas'][:, :10]
        elif 'orig_betas' in self.labels:
            self.skel_betas = self.labels['orig_betas'][:, :10]
        else: 
            raise ValueError('Unknown skel betas key')
        
        # 2dkp
        if 'gtkps' in self.labels:
            self.kp2d = self.labels['gtkps'][:,:NUM_JOINTS]
        elif 'keypoints_2d' in self.labels:
            self.kp2d = self.labels['keypoints_2d'][:,:NUM_JOINTS]
        else:
            raise ValueError('Unknown kp2d key')
        
        # 3dkp
        if 'keypoints3d' in self.labels:
            self.kp3d = self.labels['keypoints3d']
        elif 'keypoints_3d' in self.labels:
            self.kp3d = self.labels['keypoints_3d']
        else:
            raise ValueError('Unknown kped key')

        # Add confience for bedlam dataset
        if self.kp3d is not None and self.kp3d.shape[2] < 4:
            self.kp3d = np.concatenate((self.kp3d, np.ones((self.kp3d.shape[0], self.kp3d.shape[1], 1))), axis=2)
            logger.info(f'Add confience, kp3d shape: {self.kp3d.shape}')

        # cam int
        if 'cam_int' in self.labels:
            self.cam_int = self.labels['cam_int']
        else:
            raise ValueError('the cam_int is None')
        
        self.length = self.labels['scale'].shape[0]
        logger.info(f'Loaded {self.ds_name} dataset, num samples {self.length}')
         

    def prepocess_path(self, img_dir, imgname):
      
        if 'insta' in self.ds_name:
            parts = imgname.split(os.sep)
            segments = parts[2].split('_')
            parts[2] = f"{segments[0]}_{segments[1][:2]}_{segments[2][:2]}_{segments[-1]}"
            name, ext = os.path.splitext(parts[-1])
            parts[-1] = f"{int(name):04d}{ext}"
            imgname = os.sep.join(parts)
            imgname = os.path.join(img_dir, imgname.replace('insta-train/',"", 1))
        elif 'aic' in self.ds_name:
            imgname = imgname.replace('aic-train/',"", 1)
            imgname = os.path.join(img_dir, imgname)
        elif 'mpii' in self.ds_name or 'coco' in self.ds_name:
            imgname = os.path.basename(imgname)
            imgname = os.path.join(img_dir, imgname)
        elif 'h36m' in self.ds_name or 'mpi' in self.ds_name:
            imgname = imgname + '.jpg'
            imgname = os.path.join(img_dir, imgname)
        else:
            # bedlam & agora dataset
            imgname = imgname
            imgname = os.path.join(img_dir, imgname)

        return imgname 

    def __getitem__(self, index):
        item = {}
        
        imgname = self.imgname[index] # (,)
        scale = self.scale[index]  # (N,) or (N,2)
        center = self.center[index] # (N,) or (N,2)
        kp2d = self.kp2d[index][:,:NUM_JOINTS]
        kp3d = self.kp3d[index]

        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=None).max()

        if np.any(bbox_size < 1):
            raise ValueError(f"Invalid bbox size: {bbox_size}. It must be >= 1")
            
        augm_config = copy.deepcopy(self.cfg.datasets.config)

        imgname = self.prepocess_path(self.img_dir, str(imgname))
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]

        _, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        img_patch_rgba, \
        _, keypoints_2d, img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(
                                      img_path=imgname,
                                      center_x=center[0], center_y=center[1],
                                      # from scale 
                                      width=bbox_size, height=bbox_size,
                                      keypoints_2d=kp2d,
                                      # patch size
                                      patch_width=self.img_size, patch_height=self.img_size,
                                      mean=self.mean, std=self.std, 
                                      do_augment=self.is_train, augm_config=augm_config,
                                      is_bgr=True, return_trans=True,
                                      use_skimage_antialias=False,
                                      border_mode = self.border_mode,
                                      ds_name = self.ds_name
                            )
        
        item.update({
            'img': img_patch_rgba[:3],  # remove alpha
            'kp2d': keypoints_2d.astype(np.float32),
            'orig_kp2d': kp2d.astype(np.float32),
            'kp3d': kp3d.astype(np.float32),
            'box_center': np.array([cx, cy]),
            'bbox_size' : bbox_w * scale_aug,
            'img_size': np.array([img_size[0], img_size[1]]), # img_full_size, img_height, img_width
            'skel_poses': self.skel_poses[index],
            'skel_betas': self.skel_betas[index],
            'ds_name': self.ds_name,
            'cam_int' : np.array(self.cam_int[index]).astype(np.float32),
            'img_full_resized' : img_full_resized,
            '_trans' : trans
        })

        # ================= [新增 Bio-OTSR Scalar GT 逻辑] =================
        skel_pose_vec = self.skel_poses[index] # (46,)

        # 1. 提取 Type D (参数化关节)
        # 直接取值
        type_d_indices = BIO_OTSR_CONFIG['TYPE_D_INDICES']
        scalar_vals_d = skel_pose_vec[type_d_indices]

        # 2. 提取 Type B (铰链关节)
        # 注意: 如果 Config 中定义了 limit 归一化，这里最好保持原始角度，
        # 让 Loss 负责归一化，或者在这里归一化。通常建议传原始值，在 Loss 中处理。
        # 这里演示直接提取原始值。
        type_b_vals = []
        # 注意：dict 遍历顺序在不同 python 版本可能不同，建议把 TYPE_B 改为 List 结构或排好序
        # 假设 TYPE_B 是 {'knee_r': {'param': 6...}, ...}
        # 为了保证顺序一致，建议在 Config 中定义一个有序列表，或者这里按 param index 排序
        # 简便起见，我们假设 kin_skel.py 中您会维护一个有序列表 TYPE_B_LIST
        # 这里我们手动按 param index 排序提取，确保与 Decoder 输出顺序一致
        
        # 临时排序逻辑 (建议完善 kin_skel 的定义)
        type_b_items = sorted(BIO_OTSR_CONFIG['TYPE_B'].items(), key=lambda x: x[1]['param'])
        for name, info in type_b_items:
            idx = info['param']
            # 可选: 归一化到 [-1, 1] 
            # limit = info['limit']
            # val = 2 * (skel_pose_vec[idx] - limit[0]) / (limit[1] - limit[0]) - 1
            val = skel_pose_vec[idx] # 暂时取原始值
            type_b_vals.append(val)
        
        scalar_vals_b = np.array(type_b_vals, dtype=np.float32)

        # 拼接 (顺序必须与 Decoder 的 scalar_decoder 输出一致: 先 Type D 后 Type B，或反之)
        # 假设 Decoder 设计是: [Type D indices..., Type B indices...]
        item['scalar_gt'] = np.concatenate([scalar_vals_d, scalar_vals_b]).astype(np.float32)
        # ==================================================================
        return item

    def __len__(self):
        return self.length
