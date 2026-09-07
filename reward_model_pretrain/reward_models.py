import sys
from pathlib import Path

from raft_reward_utils import BrightnessConsistencyMapDiscriminator
from cotracker_core.cotracker_discriminator import TrajectoryDiscriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MotionRewardModel(nn.Module):
    def __init__(self, ic_config_dict, lc_config_dict):
        super().__init__()
        self.ic_reward_model = ICRewardModel(ic_config_dict)
        self.lc_reward_model = LCRewardModel(lc_config_dict)
        self.ic_reward_lambda = ic_config_dict['reward_weight']
        self.lc_reward_lambda = lc_config_dict['reward_weight']
    
    def forward(self, vid, lc_queries_point=None):
        ic_reward, _ = self.ic_reward_model(vid)
        lc_reward, _, lc_queries_point = self.lc_reward_model(vid, lc_queries_point)
        ic_reward = ic_reward * self.ic_reward_lambda
        lc_reward = lc_reward * self.lc_reward_lambda
        ave_reward = (ic_reward + lc_reward) / 2.0
        return ic_reward, lc_reward, ave_reward, lc_queries_point
    
    def freeze_params(self):
        self.ic_reward_model.freeze_params()
        self.lc_reward_model.freeze_params()
    
    def unfreeze_params(self):
        self.ic_reward_model.unfreeze_params()
        self.lc_reward_model.unfreeze_params()
    
    def get_trainable_params(self):
        params = []
        params.extend(self.ic_reward_model.get_trainable_params())
        params.extend(self.lc_reward_model.get_trainable_params())
        return params

class ICRewardModel(nn.Module):
    def __init__(self, config_dict):
        """
        Args:
            config_dict: the config_dict for the brightness_map_disc
        """
        super().__init__()
        # support switching + keep backward compatibility
        self.encoder_type = (config_dict.get("encoder_type", None) or "resnet").lower()
        if "resnet_discriminator_config" in config_dict or "vit_discriminator_config" in config_dict:
            if self.encoder_type == "resnet":
                arch_config = config_dict["resnet_discriminator_config"]
            elif self.encoder_type == "vit":
                arch_config = config_dict["vit_discriminator_config"]
            else:
                raise ValueError(f"Unknown encoder_type={self.encoder_type}. Use 'resnet' or 'vit'.")
        else:
            # old style config
            arch_config = config_dict["discriminator_config"]

        # initialize the brightness_map_disc
        self.brightness_map_disc = BrightnessConsistencyMapDiscriminator(
            encoder_type=self.encoder_type,
            **arch_config
        )

        # load the pretrained checkpoint
        pretrained_ckpt = config_dict['pretrained_ckpt']
        if pretrained_ckpt is not None:
            self.brightness_map_disc = self.load_pretrained_discriminator(self.brightness_map_disc, pretrained_ckpt)
        
        # freeze all the parameters of the brightness_map_disc
        self.freeze_params()
        # self.target_frame_size = arch_config['frame_size']
        self.sigmoid_squeeze_reward = config_dict['sigmoid_squeeze_reward']

    # @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self, vid):
        """
        Args:
            vid: the video to be evaluated, shape: (B, T, C, H, W), with pixel value range [0, 1]
        Returns:
            r: the reward, shape: (B,)
            out: the logits, shape: (B,)
        """
        # assert that x is in range of [0, 1]
        assert vid.min() >= 0 and vid.max() <= 1, f"input video for cotracker is not in range of [0, 1], min: {vid.min()}, max: {vid.max()}"
        # convert x to range of [0, 255]
        vid = vid * 255

        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = self.brightness_map_disc(vid)
        if self.sigmoid_squeeze_reward:
            r = F.sigmoid(out).squeeze(-1)
        else:
            r = out.squeeze(-1)
        
        return r.detach(), out.squeeze(-1)


    def load_pretrained_discriminator(self, model, pretrained_ckpt):
        pretrained_ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        pretrained_ckpt = pretrained_ckpt['discriminator_state_dict']
        # optimizer_state_dict = pretrained_ckpt['optimizer_state_dict']
        # global_step = pretrained_ckpt['global_step']
        # lr_scheduler_state_dict = pretrained_ckpt['lr_scheduler_state_dict']
        # latest_losses = pretrained_ckpt['latest_losses']

        msg = model.load_state_dict(pretrained_ckpt, strict=True)
        print('Loading pretrained checkpoint: ', msg)

        return model
    
    def freeze_params(self):
        self.brightness_map_disc.eval()
        for param in self.brightness_map_disc.parameters():
            param.requires_grad = False
    
    def unfreeze_params(self):
        if self.encoder_type == "vit":
            self.brightness_map_disc.flow_vit.train()
            key = "flow_vit"
        else:
            self.brightness_map_disc.flow_resent.train()
            key = "flow_resent"

        for name, p in self.brightness_map_disc.named_parameters():
            if key in name:
                p.requires_grad = True
    
    def get_trainable_params(self):
        key = "flow_vit" if self.encoder_type == "vit" else "flow_resent"
        return [p for n, p in self.brightness_map_disc.named_parameters() if key in n]
    
    def save_discriminator(self, path):
        # TO DO
        raise NotImplementedError("Saving the discriminator is not implemented yet")

class LCRewardModel(nn.Module):
    def __init__(self, config_dict):
        """
        Args:
            config_dict: the config_dict for the traj_discriminator
        """
        super().__init__()
        arch_config = config_dict['discriminator_config']
        # initialize the traj_discriminator
        # arch_config['train_updateformer'] = True #the custom updateformer is different from the one in the cotracker
        self.traj_discriminator = TrajectoryDiscriminator(**arch_config)

        # load the pretrained checkpoint
        pretrained_ckpt = config_dict['pretrained_ckpt']
        if pretrained_ckpt is not None:
            self.traj_discriminator = self.load_pretrained_discriminator(self.traj_discriminator, pretrained_ckpt)
        
        # freeze all the parameters of the traj_discriminator
        self.freeze_params()

        # self.target_frame_size = arch_config['frame_size']
        self.sigmoid_squeeze_reward = config_dict['sigmoid_squeeze_reward']
    
    def forward(self, vid, queries=None):
        """
        Args:
            vid: the video to be evaluated, shape: (B, T, C, H, W), with pixel value range [0, 1]
        Returns:
            r: the reward, shape: (B,)
            out: the logits, shape: (B,)
        """
        assert vid.min() >= 0 and vid.max() <= 1, f"input video for cotracker is not in range of [0, 1], min: {vid.min()}, max: {vid.max()}"
        out, queries = self.traj_discriminator.cotracker_forward(vid, queries_point=queries)
        if self.sigmoid_squeeze_reward:
            r = F.sigmoid(out).squeeze(-1)
        else:
            r = out.squeeze(-1)
        
        return r.detach(), out.squeeze(-1), queries

    def load_pretrained_discriminator(self, model, pretrained_ckpt):
        pretrained_ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        pretrained_ckpt = pretrained_ckpt['discriminator_state_dict']
        # state_dict = {}
        # import pdb; pdb.set_trace()
        # for k, v in pretrained_ckpt.items():
        #     state_dict['track_former.' + k] = v
        # optimizer_state_dict = pretrained_ckpt['optimizer_state_dict']
        # global_step = pretrained_ckpt['global_step']
        # lr_scheduler_state_dict = pretrained_ckpt['lr_scheduler_state_dict']
        # latest_losses = pretrained_ckpt['latest_losses']

        msg = model.track_former.load_state_dict(pretrained_ckpt, strict=True)
        print('Loading pretrained checkpoint: ', msg)

        return model
    
    def freeze_params(self):
        # LC reward inference mode:
        # - freeze every parameter
        # - keep the entire trajectory discriminator in eval mode
        self.traj_discriminator.eval()
        for param in self.traj_discriminator.parameters():
            param.requires_grad = False
    
    def unfreeze_params(self):
        # LC reward GAN-training mode:
        # - keep CoTracker (including its updateformer) frozen + eval
        # - train ONLY the appended TrackFormer head
        self.traj_discriminator.eval()
        for p in self.traj_discriminator.parameters():
            p.requires_grad = False

        self.traj_discriminator.track_former.train()
        for p in self.traj_discriminator.track_former.parameters():
            p.requires_grad = True
    
    def get_trainable_params(self):
        params = []
        for param in self.traj_discriminator.track_former.parameters():
            params.append(param)
        return params
    
    def save_discriminator(self, path):
        # TO DO
        raise NotImplementedError("Saving the discriminator is not implemented yet")