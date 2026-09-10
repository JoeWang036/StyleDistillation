import PIL
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor
from torchvision import transforms
from packaging import version

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

class DistilllationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_path,
        content_ref,
        style_ref,
        tokenizer_1,
        tokenizer_2,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        center_crop=False,
    ):
        self.image_path = image_path
        self.content_ref = content_ref
        self.style_ref = style_ref
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.flip_p = flip_p

        self._length = repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.clip_image_processor = CLIPImageProcessor()
        self.clip_image = None

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["original_size"] = (image.height, image.width)

        image = image.resize((self.size, self.size), resample=self.interpolation)

        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.crop(image)
        else:
            y1, x1, h, w = self.crop.get_params(image, (self.size, self.size))
            image = transforms.functional.crop(image, y1, x1, h, w)

        example["crop_top_left"] = (y1, x1)

        example["input_ids_1"] = self.tokenizer_1(
            self.content_ref,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_1.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["input_ids_2"] = self.tokenizer_2(
            self.content_ref,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_2.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

    def get_sty_img_feature(self):
        if self.clip_image is None:
            image = Image.open(self.image_path)
            self.clip_image = self.clip_image_processor(images=image,
                                                        return_tensors="pt").pixel_values
        return self.clip_image
