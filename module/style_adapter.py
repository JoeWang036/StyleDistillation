# Adapted from IP-Adapter and modified for StyleDistillation.
# See asset/THIRD_PARTY_LICENSES.md (relative to the repository root).
import os

import torch
from safetensors import safe_open

from module.attn_processor import AttnProcessor2_0, DistillationAttnProcessor2_0


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class DistillationAdapterXL(torch.nn.Module):
    # clip_dim image_encoder.config.projection_dim
    def __init__(self, unet, clip_dim, ip_ckpt, num_tokens=4, target_blocks=["block"]):
        super().__init__()
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.target_blocks = target_blocks

        self.unet = unet
        self.set_processor()
        # image proj model
        self.image_proj_model = self.init_proj(clip_dim)
        self.load_ip_adapter()

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        if unet_added_cond_kwargs is None:
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        else:
            noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states,
                                   added_cond_kwargs=unet_added_cond_kwargs).sample

        return noise_pred

    def init_proj(self, clip_dim):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=clip_dim,
            clip_extra_context_tokens=self.num_tokens,
        )
        return image_proj_model

    def set_processor(self):
        unet = self.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                selected = False
                for block_name in self.target_blocks:
                    if block_name in name:
                        selected = True
                        break
                if selected:
                    attn_procs[name] = DistillationAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=self.num_tokens,
                    )
                else:
                    attn_procs[name] = DistillationAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        num_tokens=self.num_tokens,
                        skip=True,
                    )
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    def get_image_embeds(self, image_embeds):
        image_prompt_embeds = self.image_proj_model(image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
