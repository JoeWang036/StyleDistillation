# Third-party licenses

Original contributions to StyleDistillation are licensed under the [Apache License 2.0](../LICENSE). Third-party portions retain their respective copyrights and licenses below.

The source links identify corresponding upstream implementations, not the historical revisions originally incorporated.

## Apache-2.0 portions

The complete Apache-2.0 license is included in [../LICENSE](../LICENSE).

| Local code | Upstream reference | Retained notice |
| --- | --- | --- |
| [train.py](../train.py) | [Hugging Face Diffusers](https://github.com/huggingface/diffusers) | Copyright 2025 The HuggingFace Inc. team. All rights reserved. |
| [module/style_adapter.py](../module/style_adapter.py), [module/attn_processor.py](../module/attn_processor.py) | IP-Adapter: [adapter](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py), [attention processors](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/attention_processor.py) | [IP-Adapter Apache-2.0 license](https://github.com/tencent-ailab/IP-Adapter/blob/main/LICENSE) |
| Diffusers-derived attention processors in [module/attn_processor.py](../module/attn_processor.py) | [Diffusers attention reference](https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/attention_processor.py) | Copyright 2024 The HuggingFace Team. All rights reserved. |
| CSD utility portions in [util/utils.py](../util/utils.py), including `convert_state_dict` and `convert_weights_float` | [CSD utilities](https://github.com/learn2phoenix/CSD/blob/main/CSD/utils.py) | Copyright (c) Facebook, Inc. and its affiliates. |

The CSD utility source also credits torchvision references and [DETR utilities](https://github.com/facebookresearch/detr/blob/master/util/misc.py). Its Apache-2.0 notice is retained for the corresponding portions; CSD's root MIT license does not replace that file-specific notice.

The training, adapter, and utility code has been adapted for StyleDistillation. The attention processor file retains its existing modification notice.

## MIT portions

| Local code | Upstream reference | Copyright notice |
| --- | --- | --- |
| `CSD_CLIP` in [util/utils.py](../util/utils.py) | [CSD model](https://github.com/learn2phoenix/CSD/blob/main/CSD/model.py) | Copyright (c) 2024 learn2phoenix |
| CLIP-derived conversion logic in `convert_weights_float` in [util/utils.py](../util/utils.py) | [OpenAI CLIP conversion logic](https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py) | Copyright (c) 2021 OpenAI |
| `CrossAttention` in [module/distiller.py](../module/distiller.py) | [CompVis latent-diffusion attention](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py) | Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich |

The following MIT terms apply separately to each portion above, together with its listed copyright notice.

```text
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
