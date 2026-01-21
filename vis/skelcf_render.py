from typing import Dict
import torch 
from torch import nn
from omegaconf import DictConfig
from body_models.skel_wrapper import SKELWrapper
from models.backbones.vit import ViT
from models.cam_model.fl_net import FLNet
from models.heads.skel_decoder_base import SKELTransformerDecoderHeadBase
from util.pylogger import get_pylogger

logger = get_pylogger(__name__)



def fix_prefix_state_dict_vitposeb(st: Dict):
    new_st = {}
    for k, v in st.items():
        if k.startswith('backbone.'):
            new_k = k.replace('backbone.', '', 1)
            new_st[new_k] = v

    for k, v in new_st.items():
        logger.info(f"key: {k}, value: {v.shape}")

    return new_st


class SKELCFRender(nn.Module):

    def __init__(self, cfg:DictConfig):
        super(SKELCFRender, self).__init__()
        # SKEL_wrapper 
        
        self.cfg = cfg
        self.backbone = self.setup_backbone()
        # setup the decoder, take pose_init after encoder as input
        self.decoder = self.setup_head()
        self.cam_model = self.setup_cam_model()
        self.skel_model = SKELWrapper(**cfg.hub.body_models.skel_mix_hsmr)

    def setup_cam_model(self):

        cam_model = FLNet()
        for p in cam_model.parameters():
            p.requires_grad = False
        cam_model.eval()
        return cam_model

    def setup_backbone(self):
       
        backbone = ViT(**self.cfg.hub.backbones.vit_h)
        logger.info('VITPOSE_H initialized !!')
       
        return backbone


    def setup_head(self):
      
        skel_head = SKELTransformerDecoderHeadBase(cfg=self.cfg)
        logger.info('SKEL Transformer decoder head initialized !!!')
        
        return skel_head

    def get_trainable_parameters(self):
        return list(self.backbone.parameters()) + list(self.decoder.parameters())

    def forward_step(self, x, batch, return_vit_layers=False):
      
        B = x.shape[0]
        # B, 192, 1280
        if return_vit_layers:
            # 返回所有中间层特征用于可视化
            x_norm_patch, vit_intermediate = self.backbone(x[:, :, :, 32:-32], return_intermediate=True)
        else:
            x_norm_patch = self.backbone(x[:, :, :, 32:-32])
            vit_intermediate = None
            
        cam_int, bbox_info = self.get_cam_intrinsic(batch)
        # x_cls = x_norm_patch[:, 0, :]
        # B, 1280
        x_cls = x_norm_patch.mean(1)
        # enc_poses ... / dec_poses
        outputs = self.decoder(x_cls, 
                            x_norm_patch=x_norm_patch, bbox_info=bbox_info)
        
        # 返回 ViT 特征用于可视化
        outputs['vit_features'] = x_norm_patch
        if return_vit_layers:
            outputs['vit_intermediate_layers'] = vit_intermediate
         
        return outputs, cam_int

    def get_cam_intrinsic(self, batch):
       
        
        img_h, img_w = batch['img_size'][:, 0], batch['img_size'][:, 1]  
   
        cam_intrinsics = batch['cam_int']
         
        # Original
        b = batch['bbox_size']
        cx, cy = batch['box_center'][:, 0], batch['box_center'][:, 1]
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b],
                                dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)   # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] / cam_intrinsics[:, 0, 0])  # [-1, 1]

        bbox_info = bbox_info.float().to(img_h.device)
        return cam_intrinsics, bbox_info

    def get_pd_params(self, pd_poses, pd_betas, pd_cam, batch):

        skel_params = {
            'poses': pd_poses,
            'betas': pd_betas
        }
        
        # Get the 3D keypoints, skin vertices and 2D keypoints
        skel_outputs = self.skel_model(**skel_params, skelmesh=False)
        pd_skin_verts = skel_outputs.skin_verts.detach().cpu().clone()  # (B, V=6890, 3)

        total_predict = {
            'pd_cam_t' : pd_cam,
            'pd_skin_verts' : pd_skin_verts,
            'pd_skel_params' : skel_params,
        }

        return total_predict
    

    def forward(self, batch, return_vit_layers=False):
        img_patch = batch['img']  # (B, C, H, W)

        outputs, cam_int = self.forward_step(img_patch, batch, return_vit_layers=return_vit_layers)  # {...}
        predict_enc = self.get_pd_params(outputs['pd_enc_poses'], outputs['pd_enc_betas'], outputs['pd_enc_cam'], batch)
        predict_dec = self.get_pd_params(outputs['pd_dec_poses'], outputs['pd_dec_betas'], outputs['pd_dec_cam'], batch)
        # for vis
        per_layer_params = outputs['per_layer_params']
        
        # 返回 ViT 特征用于可视化
        vit_features = outputs.get('vit_features', None)
        vit_layers = outputs.get('vit_intermediate_layers', None)
        
        # 返回 Decoder 每层特征和attention用于可视化
        decoder_features = outputs.get('per_layer_features', None)
        decoder_attentions = outputs.get('per_layer_attentions', None)
        
        return predict_enc, predict_dec, per_layer_params, vit_features, vit_layers, decoder_features, decoder_attentions

