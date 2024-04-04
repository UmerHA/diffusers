# Copyright 2024 The HuggingFace Team. All rights reserved.
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

# Based on the paper "Tuning-Free Noise Rectification for High Fidelity Image-to-Video Generation" by Weijie Li et al. (https://arxiv.org/abs/2403.02827)

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline
from diffusers.pipelines.animatediff.pipeline_output import AnimateDiffPipelineOutput
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# copied from pipelines.animatediff.pipeline_animatediff
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


def preprocess_image(image, width, height):
    assert isinstance(image, PIL.Image.Image)
    image = np.array(image.resize((width, height))).astype(np.float32) / 255.0
    image = np.expand_dims(image, 0)
    image = image.transpose(0, 3, 1, 2)
    image = 2.0 * image - 1.0
    image = torch.from_numpy(image)
    return image


class NoiseRectifiedAnimatePipeline(AnimateDiffPipeline):
    r"""
    Pipeline for text-to-video generation with noise-rectification.

    This model inherits from [`AnimateDiffPipeline`], which in turn inherits from [`DiffusionPipeline`]. Check the superclass documentations for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A [`UNet2DConditionModel`] used to create a UNetMotionModel to denoise the encoded video latents.
        motion_adapter ([`MotionAdapter`]):
            A [`MotionAdapter`] to be used in combination with `unet` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def prepare_latents(
        self,
        input_image,
        batch_size,
        num_channels_latents,
        num_frames,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            if isinstance(generator, list):
                latents = [
                    torch.randn(shape, generator=generator[i], device=device, dtype=dtype) for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # Noise rectification first adds noise to the input image, but also keeps the added noise for latter rectification.
        noise = latents.clone()
        if input_image is not None:
            input_image = preprocess_image(input_image, width, height).to(device=device, dtype=dtype)

            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(input_image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(input_image).latent_dist.sample(generator)
        else:
            init_latents = None

        if init_latents is not None:
            # einops.rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size, f=1)
            init_latents = init_latents.view(batch_size, -1, 1, init_latents.size(2), init_latents.size(3))
            init_latents = init_latents.repeat((1, 1, num_frames, 1, 1)) * 0.18215

        noisy_latents = self.scheduler.add_noise(init_latents, noise, self.scheduler.timesteps[0])
        return noisy_latents, noise

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_frames: Optional[int] = 16,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        input_image=None,
        noise_rectification_start: float = 0.0,
        noise_rectification_end: float = 1.0,
        noise_rectification_weight: Optional[torch.Tensor] = None,
        noise_rectification_weight_start_omega=1.0,
        noise_rectification_weight_end_omega=0.5,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        callback_steps = None
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents, noise = self.prepare_latents(
            input_image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # rectify predicted noise with the ground truth noise
                if noise_rectification_weight is None:
                    noise_rectification_weight = torch.cat(
                        [
                            torch.linspace(
                                noise_rectification_weight_start_omega,
                                noise_rectification_weight_end_omega,
                                num_frames // 2,
                            ),
                            torch.linspace(
                                noise_rectification_weight_end_omega,
                                noise_rectification_weight_end_omega,
                                num_frames // 2,
                            ),
                        ]
                    )
                noise_rectification_weight = noise_rectification_weight.view(1, 1, num_frames, 1, 1)
                noise_rectification_weight = noise_rectification_weight.to(latent_model_input.dtype).to(
                    latent_model_input.device
                )

                if noise_rectification_start <= i / len(timesteps) < noise_rectification_end:
                    delta_frames = noise - noise_pred
                    delta_noise_adjust = (
                        noise_rectification_weight * (delta_frames[:, :, [0], :, :].repeat((1, 1, num_frames, 1, 1)))
                        + (1 - noise_rectification_weight) * delta_frames
                    )
                    noise_pred = noise_pred + delta_noise_adjust

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents)
            video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # 10. Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return AnimateDiffPipelineOutput(frames=video)
