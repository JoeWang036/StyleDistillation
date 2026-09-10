# Includes code from CSD and OpenAI CLIP, combined and modified for StyleDistillation.
# See asset/THIRD_PARTY_LICENSES.md (relative to the repository root) for licenses.
import copy
from collections import OrderedDict

import clip
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict


def convert_weights_float(model: nn.Module):
    """Convert applicable model parameters to fp32"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


class CSD_CLIP(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='vit_large',content_proj_head='default', model_path=None, device=None):
        super(CSD_CLIP, self).__init__()
        self.content_proj_head = content_proj_head
        if name == 'vit_large':
            if model_path is None:
                raise ValueError("model_path is required")
            else:
                clipmodel, _ = clip.load(model_path, device=device if device is not None else 'cuda')
            self.backbone = clipmodel.visual
            self.embedding_dim = 1024
        else:
            raise Exception('This model is not implemented')

        convert_weights_float(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        self.backbone.proj = None

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, input_data):
        
        feature = self.backbone(input_data)

        reverse_feature = feature

        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)

        content_output = reverse_feature @ self.last_layer_content
        content_output = nn.functional.normalize(content_output, dim=1, p=2)
        return feature, content_output, style_output


def prepare_csd(csd_clip_model_path, csd_checkpoint_path, device):
    # init model
    model = CSD_CLIP("vit_large", "default", model_path=csd_clip_model_path, device=device)
    # load model
    checkpoint = torch.load(csd_checkpoint_path, map_location="cpu")
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # image preprocess
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    preprocess = transforms.Compose([
        transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return model, preprocess


def prepare_clip(clip_model_path, device):
    model, preprocess = clip.load(clip_model_path, device=device)
    return model, preprocess


def extract_clip_features(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


@torch.no_grad()
def calculate_clip_text_score(prompt, image_path, model, preprocess, device):
    text = clip.tokenize([prompt]).to(device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    txt_features = extract_clip_features(model, text, 'txt')
    img_features = extract_clip_features(model, image, 'img')

    # normalize features
    txt_features = txt_features / txt_features.norm(dim=1, keepdim=True).to(torch.float32)
    img_features = img_features / img_features.norm(dim=1, keepdim=True).to(torch.float32)

    logit_scale = model.logit_scale.exp()
    score = logit_scale * (img_features * txt_features).sum()
    return score


@torch.no_grad()
def calculate_csd_score(
    reference_image_path,
    image_path,
    model,
    preprocess,
    device,
):
    reference_image = preprocess(Image.open(reference_image_path)).unsqueeze(0).to(device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    _, _, reference_style_output = model(reference_image)
    _, _, style_output = model(image)
    ori_features = reference_style_output.cpu().numpy()
    tar_features = style_output.cpu().numpy()

    full_sim = ori_features @ tar_features.T
    sim = full_sim.diagonal()
    return sim.mean()
