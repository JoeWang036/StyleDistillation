import argparse

from util.utils import (
    calculate_clip_text_score,
    calculate_csd_score,
    prepare_clip,
    prepare_csd,
)


def parse_args():
    parser = argparse.ArgumentParser("StyleDistillation metrics")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--reference_image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--clip_model_path", type=str, required=True)
    parser.add_argument("--csd_clip_model_path", type=str, required=True)
    parser.add_argument("--csd_checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main(args):
    csd_model, csd_preprocess = prepare_csd(
        args.csd_clip_model_path,
        args.csd_checkpoint_path,
        args.device,
    )
    clip_model, clip_preprocess = prepare_clip(args.clip_model_path, args.device)

    clip_text_score = calculate_clip_text_score(
        args.prompt,
        args.image_path,
        clip_model,
        clip_preprocess,
        args.device,
    )
    csd_score = calculate_csd_score(
        args.reference_image_path,
        args.image_path,
        csd_model,
        csd_preprocess,
        args.device,
    )

    print(f"CLIPText: {clip_text_score.item()}")
    print(f"CSD: {csd_score.item()}")


if __name__ == "__main__":
    main(parse_args())
