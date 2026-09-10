#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for StyleDistillation. See asset/THIRD_PARTY_LICENSES.md.
import time

import argparse
import logging
import math
import os
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler

from util.dataset import DistilllationDataset
from module.style_adapter import DistillationAdapterXL
from module.distiller import DistillationModule_v2

# ------------------------------------------------------------------------------

logger = get_logger(__name__)


def save_model(
    style_extractor,
    prefix,
    output_dir,
):

    prefix = f"{prefix:04d}" if isinstance(prefix, int) else prefix
    torch.save(style_extractor.state_dict(), os.path.join(output_dir, f"{prefix}.bin"))

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # loss weight pre_loss  orth_loss  content_loss  style_loss
    parser.add_argument(
        "--rebuild_scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--pre_loss_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--orth_loss_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--content_anchor_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--style_anchor_weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--rebuild_loss_weight",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--ip_ckpt",
        type=str,
        default=None,
        required=True,
        help="Path to ip-adapter checkpoint",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to the VAE model directory.",
    )
    parser.add_argument(
        "--train_img_path", type=str, default=None, required=True, help="style image path"
    )
    parser.add_argument(
        "--content_ref",
        type=str,
        required=True,
        help="content text",
    )
    parser.add_argument(
        "--style_ref", type=str, required=True, help="style text"
    )
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=300,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=10, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    args = parser.parse_args()

    return args


def encode_prompts_pool(prompts, tokenizer, text_encoder):
    input_id = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    encoder_output_pool = text_encoder(input_id.to(text_encoder.device).unsqueeze(0), output_hidden_states=True)[0]
    return encoder_output_pool

def anchor_clip(vec, ref_vec):

    cosine_sim = F.cosine_similarity(vec, ref_vec, dim=1)

    clip_score = torch.clamp(cosine_sim, min=0.0)
    return clip_score

def orthogonality_loss(c, s):

    c_norm = c / c.norm(dim=1, keepdim=True)
    s_norm = s / s.norm(dim=1, keepdim=True)

    clip_score = torch.clamp((c_norm * s_norm).sum(dim=1), min=0.0)
    return clip_score

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    vae = AutoencoderKL.from_pretrained(
        args.vae_path
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # Load the image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # Load Adapter ["block"] ["up_blocks"]
    style_adapter = DistillationAdapterXL(unet, image_encoder.config.projection_dim,
                                          args.ip_ckpt, num_tokens=4, target_blocks=["up_blocks"])

    # Freeze vae and unet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    style_adapter.requires_grad_(False)

    # Dataset and DataLoaders creation:
    train_dataset = DistilllationDataset(
        args.train_img_path,
        args.content_ref,
        args.style_ref,
        tokenizer_1,
        tokenizer_2,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    content_anchor = encode_prompts_pool(args.content_ref, tokenizer_2, text_encoder_2)
    style_anchor = encode_prompts_pool(args.style_ref, tokenizer_2, text_encoder_2)

    anchor_image_embed = image_encoder(
        train_dataset.get_sty_img_feature()).image_embeds
    image_patch_embed = image_encoder(
        train_dataset.get_sty_img_feature()).last_hidden_state
    print(image_patch_embed.shape)


    style_extractor = DistillationModule_v2(
        dim_t=image_encoder.config.projection_dim,
        dim_i=image_encoder.config.projection_dim,
        n_heads=8,
        d_head=64,
        style_emb=style_anchor,
        content_emb=content_anchor,
        image_emb=anchor_image_embed,
        image_emb_patch=image_patch_embed,
        dropout=0.05,
    )

    # Initialize the optimizer
    optimizer_class = torch.optim.AdamW

    params_to_opt = style_extractor.parameters()
    optimizer = optimizer_class(
        # only optimize the embeddings
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    style_extractor.train()
    # Prepare everything with our `accelerator`.
    style_extractor, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        style_extractor, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet and text_encoder_2 to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)  # use fp32
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    style_adapter.to(accelerator.device, dtype=weight_dtype)
    content_anchor = content_anchor.to(accelerator.device, dtype=weight_dtype)
    style_anchor = style_anchor.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # Train!

    start_time = time.time()

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    anchor_image_embed = anchor_image_embed.to(accelerator.device, dtype=weight_dtype)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(style_extractor):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states_1 = (
                    text_encoder_1(batch["input_ids_1"], output_hidden_states=True)
                    .hidden_states[-2]
                    .to(dtype=weight_dtype)
                )
                encoder_output_2 = text_encoder_2(batch["input_ids_2"], output_hidden_states=True)
                encoder_hidden_states_2 = encoder_output_2.hidden_states[-2].to(dtype=weight_dtype)
                original_size = [
                    (batch["original_size"][0][i].item(), batch["original_size"][1][i].item())
                    for i in range(args.train_batch_size)
                ]
                crop_top_left = [
                    (batch["crop_top_left"][0][i].item(), batch["crop_top_left"][1][i].item())
                    for i in range(args.train_batch_size)
                ]
                target_size = (args.resolution, args.resolution)
                add_time_ids = torch.cat(
                    [
                        torch.tensor(original_size[i] + crop_top_left[i] + target_size)
                        for i in range(args.train_batch_size)
                    ]
                ).to(accelerator.device, dtype=weight_dtype)
                added_cond_kwargs = {"text_embeds": encoder_output_2[0], "time_ids": add_time_ids}
                encoder_hidden_states = torch.cat([encoder_hidden_states_1, encoder_hidden_states_2], dim=-1)

                sty_embed, content_embed = style_extractor()  # 1 1280
                image_embeds = sty_embed.repeat(bsz, 1).to(dtype=weight_dtype)
                # Predict the noise residual
                model_pred = style_adapter(noisy_latents, timesteps,
                                           encoder_hidden_states, added_cond_kwargs, image_embeds)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                pre_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                pre_loss = pre_loss * args.pre_loss_weight
                # content_anchor
                content_anchor_loss = 1 - anchor_clip(
                    content_embed, content_anchor
                )
                content_anchor_loss = content_anchor_loss * args.content_anchor_weight

                orth_loss = orthogonality_loss(content_embed, sty_embed)
                orth_loss = orth_loss * args.orth_loss_weight
                # style_anchor
                style_anchor_loss = F.mse_loss(
                    sty_embed.to(dtype=weight_dtype).float(), anchor_image_embed.float(), reduction="mean"
                )
                style_anchor_loss = style_anchor_loss * args.style_anchor_weight
                # style_target
                sty_embed_norm = sty_embed / sty_embed.norm(dim=1, keepdim=True)
                content_embed_norm = content_embed / content_embed.norm(dim=1, keepdim=True)

                merge_emb = sty_embed_norm + content_embed_norm * args.rebuild_scale
                style_loss = 1 - anchor_clip(
                    merge_emb.to(dtype=weight_dtype), anchor_image_embed
                )
                style_loss = style_loss * args.rebuild_loss_weight

                loss = pre_loss + orth_loss + content_anchor_loss + style_anchor_loss + style_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_opt, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    print(f"Saving model checkpoint to {args.output_dir}")
                    checkpoint = accelerator.unwrap_model(style_extractor)
                    save_model(
                        checkpoint,
                        global_step,
                        args.output_dir,
                    )

            logs = {"loss": loss.detach().item(),
                    "pre_loss": pre_loss.detach().item(),
                    "orth_loss": orth_loss.detach().item(),
                    "content_anchor_loss": content_anchor_loss.detach().item(),
                    "style_anchor_loss": style_anchor_loss.detach().item(),
                    "style_loss": style_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Training completed in {elapsed_time:.2f} seconds.")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Saving model checkpoint to {args.output_dir}")
        checkpoint = accelerator.unwrap_model(style_extractor)
        save_model(
            checkpoint,
            "final",
            args.output_dir,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
