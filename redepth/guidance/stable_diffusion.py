import torch
import torch.nn.functional as F

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)


class StableDiffusionPromptProcessor:

    def __init__(self, guidance_model):
        self.guidance_model = guidance_model
        self.device = self.guidance_model.device

    @torch.no_grad()
    def encode_prompts(self, prompt, negative_prompt=""):
        text_intput = self.guidance_model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.guidance_model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_emb = self.guidance_model.text_encoder(
            text_intput.input_ids.to(self.device)
        )[0]

        # Do the same for unconditional embeddings
        uncond_input = self.guidance_model.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.guidance_model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_emb = self.guidance_model.text_encoder(
            uncond_input.input_ids.to(self.guidance_model.text_encoder.device)
        )[0]

        # Cat for final embeddings
        text_emb = torch.cat([text_emb, uncond_emb])
        return text_emb


class StableDiffusionGuidance(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": "./.cache",
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.requires_grad_(False)

        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.eval()

        if self.cfg.scheduler == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="scheduler",
                torch_dtype=self.weights_dtype,
            )
        else:
            raise NotImplementedError(f"Scheduler {self.cfg.scheduler} not implemented")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.step = 0
        self.max_steps = self.cfg.max_steps
        self.set_min_max_steps()
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    def set_min_max_steps(self):
        min_step_percent = self.cfg.min_step_percent
        max_step_percent = self.cfg.max_step_percent
        self.min_t_step = int(self.num_train_timesteps * min_step_percent)
        self.max_t_step = int(self.num_train_timesteps * max_step_percent)

    def forward_unet(
        self,
        latents,
        t,
        encoder_hidden_states,
    ):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def encode_images(self, imgs):
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    def decode_latents(
        self,
        latents,
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=True
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def compute_grad_sds(
        self,
        latents,
        t,
        prompt_embedding,
    ):
        batch_size = latents.shape[0]

        with torch.no_grad():
            # add noise
            noise = torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
            )
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=prompt_embedding,
            )

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)

        return grad

    def forward(
        self,
        rgb,
        prompt_embedding,
        rgb_as_latents=False,
    ):
        bs = rgb.shape[0]

        if not rgb_as_latents:
            rgb_BCHW_512 = self.resize(rgb, size=512, fill=0)
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        else:
            latents = F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=True)

        t = torch.randint(
            self.min_t_step,
            self.max_t_step + 1,
            [bs],
            dtype=torch.long,
            device=self.device,
        )

        grad = self.compute_grad_sds(latents, t, prompt_embedding)

        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()

        loss_sds = 0.5 * F.mse_loss(latents, target)

        return loss_sds

    def resize(self, rgb, size=512, fill=0):
        _, _, h, w = rgb.shape
        scale = min(size / h, size / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        rgb_resized = F.interpolate(
            rgb, size=(new_h, new_w), mode="bilinear", align_corners=True
        )

        pad_h = size - new_h
        pad_w = size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        rgb_padded = F.pad(
            rgb_resized, (pad_left, pad_right, pad_top, pad_bottom), value=fill
        )

        return rgb_padded
