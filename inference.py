import os

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import argparse
from torch import nn
from transformers import CLIPVisionModelWithProjection

from module.style_adapter import DistillationAdapterXL
from module.distiller import DistillationModule_v2


def prepare_style_emb(image_encoder_path, ip_ckpt, unet, style_path):
    # Load the image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    # Load Adapter ["block"]
    style_adapter = DistillationAdapterXL(unet, image_encoder.config.projection_dim,
                                          ip_ckpt, num_tokens=4, target_blocks=["up_blocks"])


    style_path = os.path.join(style_path, "final.bin")
    style_extractor = DistillationModule_v2(
        dim_t=image_encoder.config.projection_dim,
        dim_i=image_encoder.config.projection_dim,
        n_heads=8,
        d_head=64,
        style_emb=nn.Parameter(torch.zeros(1, image_encoder.config.projection_dim),
                               requires_grad=False),
        content_emb=nn.Parameter(torch.zeros(1, image_encoder.config.projection_dim),
                                 requires_grad=False),
        image_emb=nn.Parameter(torch.zeros(1, image_encoder.config.projection_dim),
                               requires_grad=False),
        image_emb_patch=nn.Parameter(torch.zeros(1, 257, 1664),
                                     requires_grad=False),
        dropout=0.05,
    )
    style_extractor.load_state_dict(torch.load(style_path))

    return style_extractor, style_adapter


def main(args):
    # first
    # third
    base_model_path = args.pretrained_model_name_or_path

    image_encoder_path = args.image_encoder_path
    ip_ckpt = args.ip_ckpt
    device = "cuda"

    os.makedirs(args.save_root, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        args.vae_path,
        torch_dtype=torch.float16,
    )
    # load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        vae=vae,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    # reduce memory consumption
    pipe.enable_vae_tiling()

    style_extractor, style_adapter = prepare_style_emb(image_encoder_path, ip_ckpt, pipe.unet,
                                                       args.style_path)

    pipe.to(device)
    style_extractor.to(device)
    style_adapter.to(device, dtype=torch.float16)

    style_extractor.eval()
    style_adapter.eval()

    seed = 42
    generator = torch.Generator("cuda").manual_seed(seed)

    prompts = [args.prompt]

    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    for prompt_id, prompt in enumerate(prompts):
        with torch.inference_mode():
            sty_emb, _ = style_extractor()
            (
                prompt_embeds_,
                negative_prompt_embeds_,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            sty_emb = sty_emb + pooled_prompt_embeds * args.add_content_scale

            sty_emb, uc_sty_emb = style_adapter.get_image_embeds(sty_emb.to(dtype=torch.float16))

            prompt_embeds = torch.cat([prompt_embeds_, sty_emb], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uc_sty_emb], dim=1)

        for idx in range(args.num_samples):
            save_path = f"{args.save_root}/{prompt_id}_{idx}.jpg"

            image = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=50,
                generator=generator).images[0]

            image.save(save_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--style_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--image_encoder_path", type=str, required=True)
    parser.add_argument("--ip_ckpt", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--add_content_scale", type=float, default=0.35)

    args = parser.parse_args()
    main(args)
