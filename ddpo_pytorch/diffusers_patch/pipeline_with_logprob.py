# Copied from https://github.com/huggingface/diffusers/blob/fc6acb6b97e93d58cb22b5fee52d884d77ce84d8/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# with the following modifications:
# - It uses the patched version of `ddim_step_with_logprob` from `ddim_with_logprob.py`. As such, it only supports the
#   `ddim` scheduler.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
# - It works for both StableDiffusionPipeline (prompt-based) and DDPMPipeline (unconditional).

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
from .ddim_with_logprob import ddim_step_with_logprob


@torch.no_grad()
def pipeline_with_logprob(
    self: StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    return_all_latents: bool = True,
):
    # ----------------------------
    # 0. Default height and width
    # ----------------------------
    if height is None:
        if hasattr(self, "vae_scale_factor") and getattr(self.unet.config, "sample_size", None) is not None:
            height = self.unet.config.sample_size * self.vae_scale_factor
        elif getattr(self.unet.config, "sample_size", None) is not None:
            height = self.unet.config.sample_size
        else:
            height = 32  # fallback (CIFAR10 default)
    if width is None:
        if hasattr(self, "vae_scale_factor") and getattr(self.unet.config, "sample_size", None) is not None:
            width = self.unet.config.sample_size * self.vae_scale_factor
        elif getattr(self.unet.config, "sample_size", None) is not None:
            width = self.unet.config.sample_size
        else:
            width = 32

    # ----------------------------
    # 1. Check inputs if SD pipeline
    # ----------------------------
    if hasattr(self, "check_inputs"):
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

    # ----------------------------
    # 2. Batch size
    # ----------------------------
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    elif prompt_embeds is not None:
        batch_size = prompt_embeds.shape[0]
    else:
        batch_size = 1

    # robust device detection for different pipeline types
    try:
        device = self._execution_device
    except Exception:
        # try pipeline.device (some pipelines expose this)
        device = getattr(self, "device", None)
        if device is None:
            # fall back to the device of the UNet parameters (most reliable)
            try:
                device = next(self.unet.parameters()).device
            except StopIteration:
                # extreme fallback
                device = torch.device("cpu")
    do_classifier_free_guidance = guidance_scale > 1.0

    # ----------------------------
    # 3. Encode input prompt (if available)
    # ----------------------------
    if hasattr(self, "_encode_prompt"):
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
    else:
        # No text encoder (DDPMPipeline case)
        prompt_embeds = torch.zeros(
            (batch_size, 1), device=device, dtype=torch.float32
        )

    # ----------------------------
    # 4. Prepare timesteps
    # ----------------------------
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # ----------------------------
    # 5. Prepare latent variables
    # ----------------------------
    num_channels_latents = self.unet.config.in_channels
    # Handle StableDiffusionPipeline vs DDPMPipeline
    if hasattr(self, "prepare_latents"):
        # StableDiffusionPipeline and other latent-diffusion pipelines
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=self.unet.dtype,
            device=self.device,
            generator=generator,
            latents=None,
        )
    else:
        # DDPMPipeline (no prepare_latents, pixel-space)
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height, width),
            device=self.device,
            dtype=self.unet.dtype,
            generator=generator,
        )

    # ----------------------------
    # 6. Prepare extra step kwargs
    # ----------------------------
    # Some pipelines (e.g., StableDiffusionPipeline) implement this helper,
    # others (e.g., DDPMPipeline) do not. Fall back to empty dict.
    if hasattr(self, "prepare_extra_step_kwargs"):
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    else:
        extra_step_kwargs = {}

    # ----------------------------
    # 7. Denoising loop
    # ----------------------------
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    all_latents = [latents]
    all_log_probs = []

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            # noise_pred = self.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=prompt_embeds if "encoder_hidden_states" in self.unet.forward.__code__.co_varnames else None,
            #     cross_attention_kwargs=cross_attention_kwargs if "cross_attention_kwargs" in self.unet.forward.__code__.co_varnames else None,
            #     return_dict=False,
            # )[0]


            import inspect

            # build kwargs for UNet forward only with supported args
            _unet_sig = None
            try:
                _unet_sig = inspect.signature(self.unet.forward)
                _unet_params = set(_unet_sig.parameters.keys())
            except Exception:
                # Fallback: try to read code varnames (older approach)
                try:
                    _unet_params = set(self.unet.forward.__code__.co_varnames)
                except Exception:
                    _unet_params = set()

            unet_call_kwargs = {}
            # only pass encoder_hidden_states if the UNet accepts it
            if "encoder_hidden_states" in _unet_params and prompt_embeds is not None:
                unet_call_kwargs["encoder_hidden_states"] = prompt_embeds
            # only pass cross_attention_kwargs if accepted and provided
            if "cross_attention_kwargs" in _unet_params and cross_attention_kwargs is not None:
                unet_call_kwargs["cross_attention_kwargs"] = cross_attention_kwargs
            # some UNets accept `return_dict` as kwarg
            if "return_dict" in _unet_params:
                unet_call_kwargs["return_dict"] = False

            # call unet with positional args + safe kwargs
            # note: latent_model_input and t are always passed positionally
            unet_out = self.unet(latent_model_input, t, **unet_call_kwargs)

            # unet may return tuple or ModelOutput; keep original behavior
            if isinstance(unet_out, tuple):
                noise_pred = unet_out[0]
            else:
                # ModelOutput-like: attempt to get .sample or first field
                noise_pred = getattr(unet_out, "sample", None)
                if noise_pred is None:
                    # fallback to indexing
                    try:
                        noise_pred = unet_out[0]
                    except Exception:
                        # last-resort: raise informative error
                        raise RuntimeError(
                            "Unable to extract noise_pred from UNet forward output. "
                            "UNet forward returned unexpected type."
                        )


            #########################################################################################

            # perform guidance (if classifier-free supported)
            if do_classifier_free_guidance and noise_pred.shape[0] % 2 == 0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                if guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob = ddim_step_with_logprob(
                self.scheduler, noise_pred, t, latents, **extra_step_kwargs
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            # callback
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # ----------------------------
    # 8. Decode or return image
    # ----------------------------
    if not output_type == "latent":
        if hasattr(self, "vae"):  # StableDiffusion case
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image = torch.nan_to_num(image.clamp(0, 1))
            if hasattr(self, "run_safety_checker"):
                image, has_nsfw_concept = self.run_safety_checker(
                    image, device, prompt_embeds.dtype
                )
            else:
                has_nsfw_concept = None
        else:  # DDPMPipeline case
            image = latents
            has_nsfw_concept = None
    else:
        image = latents
        has_nsfw_concept = None

    if has_nsfw_concept is None:
        do_denormalize = [True] * image.shape[0]
    else:
        do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

    if hasattr(self, "image_processor"):
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

    # Offload last model to CPU
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        self.final_offload_hook.offload()

    if return_all_latents:
        return image, has_nsfw_concept, all_latents, all_log_probs
    else:
        return image, has_nsfw_concept, all_latents[-1], all_log_probs[-1]
