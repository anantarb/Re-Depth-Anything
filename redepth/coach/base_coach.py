import os
import datetime
import torch
import time
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import json
from tqdm import trange
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from redepth.guidance.stable_diffusion import (
    StableDiffusionGuidance,
    StableDiffusionPromptProcessor,
)
from redepth.renderer.shaded_depth_renderer import ShadedDepthRenderer
from redepth.utils.utils import DepthEnsemble
from redepth.utils.utils import (
    linear_to_srgb,
    srgb_to_linear,
    laplacian_smoothness,
    align_least_squares,
    compute_errors,
    save_mask,
    save_RGB,
    colorize_depth,
)

dtype_mapping = {
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
    "torch.int64": torch.int64,
}


class Trainer(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = dtype_mapping[self.cfg.dtype]
        self.guidance = StableDiffusionGuidance(cfg.guidance)
        self.renderer = ShadedDepthRenderer(cfg.renderer)
        prompt = self.train_dataset.__getitem__(idx=0)["prompt"]
        self.prompt_embedding = StableDiffusionPromptProcessor(
            self.guidance
        ).encode_prompts(prompt)

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=False
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False
        )
        self.max_steps = self.cfg.max_steps
        self.ensemble_size = self.cfg.ensemble_size
        self.depth_ensemble_raw = DepthEnsemble()
        self.depth_ensemble_scaled = DepthEnsemble()

        day_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        hms_timestamp = datetime.datetime.now().strftime("%H%M%S")
        self.save_dir = os.path.join(
            cfg.logging.save_dir,
            prompt,
            day_timestamp,
            hms_timestamp,
        )
        self.save_interval = cfg.logging.save_interval

        self.step = 0
        self.writer = SummaryWriter(self.save_dir)

        with open(os.path.join(self.save_dir, "config.yaml"), "w") as fp:
            OmegaConf.save(config=cfg, f=fp)

    @abstractmethod
    def reset_depth_model(self):
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        raise NotImplementedError

    def _run_batch(self, sample):

        iter_time_start = time.time()
        self.step += 1
        self.optimizer.zero_grad()
        loss = 0
        for k, v in sample.items():
            if torch.is_tensor(v):
                sample[k] = v.to(self.device).to(self.dtype)
        raw_values, scaled_values = self.model(sample["input_image"])
        img_linear = srgb_to_linear(sample["org_image"])
        rendered_image_linear = self.renderer(scaled_values, img_linear)
        rendered_image_rgb = linear_to_srgb(rendered_image_linear)
        rendered_image_rgb = torch.clamp(rendered_image_rgb, min=1e-3, max=1.0)

        loss += self.guidance(rendered_image_rgb, self.prompt_embedding)

        self.writer.add_scalar("train/SDS loss", loss.item(), self.step)

        dmin = raw_values.clone().detach().amin(dim=(1, 2), keepdim=True)
        dmax = raw_values.clone().detach().amax(dim=(1, 2), keepdim=True)

        depth_normalized = (raw_values - dmin) / (dmax - dmin)
        smooth_loss = self.cfg.training.smoothness_weight * laplacian_smoothness(
            depth_normalized.unsqueeze(1)
        )
        loss += smooth_loss
        self.writer.add_scalar("train/Smoothness loss", smooth_loss.item(), self.step)
        self.writer.add_scalar("train/Total loss", loss.item(), self.step)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        iter_time_end = time.time()
        self.writer.add_scalar(
            "train/Time Taken per Iteration",
            iter_time_end - iter_time_start,
            self.step,
        )

    def train(self):
        self.eval()
        total_time_start = time.time()
        for N in trange(self.ensemble_size, desc="Ensemble", unit="model"):
            if N > 0:
                self.reset_depth_model()
            for _ in trange(self.max_steps, desc=f"Steps", unit="step", leave=False):
                for sample in self.train_dataloader:
                    self._run_batch(sample)
                    if self.step % self.save_interval == 0:
                        self.eval()

            if self.ensemble_size > 1:
                with torch.no_grad():
                    raw_values, scaled_values = self.model(sample["input_image"])
                    self.depth_ensemble_raw.add_depth(raw_values)
                    self.depth_ensemble_scaled.add_depth(scaled_values)

        total_time_end = time.time()
        self.writer.add_scalar(
            "train/Total Time Taken", total_time_start - total_time_end, self.step
        )

    @torch.no_grad()
    def eval(self):
        eval_image_dir = os.path.join(self.save_dir, "image")
        eval_model_dir = os.path.join(self.save_dir, "model")
        eval_depth_dir = os.path.join(self.save_dir, "depths")
        eval_normal_dir = os.path.join(self.save_dir, "normals")
        eval_stacked_dir = os.path.join(self.save_dir, "stacked")
        eval_mask_dir = os.path.join(self.save_dir, "mask")
        eval_depth_mask_dir = os.path.join(self.save_dir, "depth_mask")

        if (
            os.path.exists(eval_image_dir)
            or not os.path.exists(eval_model_dir)
            or not os.path.exists(eval_depth_dir)
            or not os.path.exists(eval_normal_dir)
            or not os.path.exists(eval_stacked_dir)
            or not os.path.exists(eval_mask_dir)
            or not os.path.exists(eval_depth_mask_dir)
        ):
            os.makedirs(eval_image_dir, exist_ok=True)
            os.makedirs(eval_model_dir, exist_ok=True)
            os.makedirs(eval_depth_dir, exist_ok=True)
            os.makedirs(eval_normal_dir, exist_ok=True)
            os.makedirs(eval_stacked_dir, exist_ok=True)
            os.makedirs(eval_mask_dir, exist_ok=True)
            os.makedirs(eval_depth_mask_dir, exist_ok=True)

        with torch.no_grad():
            for sample in self.test_dataloader:
                for k, v in sample.items():
                    if torch.is_tensor(v):
                        sample[k] = v.to(self.device).to(self.dtype)
                raw_values, scaled_values = self.model(sample["input_image"])
                if self.ensemble_size > 1 and not self.depth_ensemble_raw.isEmpty():
                    raw_values = (
                        raw_values
                        + self.depth_ensemble_raw() * self.depth_ensemble_raw.len()
                    ) / (self.depth_ensemble_raw.len() + 1)
                    scaled_values = (
                        scaled_values
                        + self.depth_ensemble_scaled()
                        * self.depth_ensemble_scaled.len()
                    ) / (self.depth_ensemble_scaled.len() + 1)

                torch.save(
                    raw_values.cpu(),
                    os.path.join(eval_depth_dir, f"{self.step}_raw.pt"),
                )
                torch.save(
                    scaled_values.cpu(),
                    os.path.join(eval_depth_dir, f"{self.step}_scaled.pt"),
                )
        if self.step == 0:
            save_RGB(sample["org_image"], eval_image_dir, "image")
            if "depth" in sample:
                torch.save(
                    sample["depth"].cpu(), os.path.join(eval_depth_dir, f"GT.pt")
                )
                save_mask(sample["depth_mask"], eval_depth_mask_dir, "mask")

            if "mask" in sample:
                save_mask(sample["mask"], eval_mask_dir, "mask")

        if "depth" in sample:
            aligned_depths = torch.zeros_like(raw_values)
            aligned_depths_vis = torch.zeros_like(raw_values)
            gt_depth_vis = torch.zeros_like(raw_values)
            errors = []
            for i in range(sample["depth"].shape[0]):
                if self.cfg.depth_model.pred_space == "disparity":
                    gt_disparity = torch.zeros_like(sample["depth"][i])
                    gt_disparity[sample["depth"][i] > 0] = (
                        1 / sample["depth"][i][sample["depth"][i] > 0]
                    )

                    scale, shift = align_least_squares(
                        gt_disparity, raw_values[i], sample["depth_mask"][i]
                    )
                    aligned_disp = raw_values[i] * scale + shift

                    aligned_depth = torch.zeros_like(aligned_disp)
                    aligned_depth[aligned_disp > 0] = 1 / aligned_disp[aligned_disp > 0]

                scale, shift = align_least_squares(
                    sample["depth"][i], aligned_depth, sample["depth_mask"][i]
                )
                aligned_depths[i, :, :] = aligned_depth * scale + shift
                errors.append(
                    compute_errors(
                        sample["depth"][i][sample["depth_mask"][i].bool()]
                        .cpu()
                        .numpy(),
                        aligned_depths[i][sample["depth_mask"][i].bool()].cpu().numpy(),
                    )
                )
                lo = torch.quantile(
                    sample["depth"][i][sample["depth_mask"][i].bool()], 0.1
                )
                hi = torch.quantile(
                    sample["depth"][i][sample["depth_mask"][i].bool()], 0.99
                )

                gt_depth_vis[i, :, :] = torch.clamp(
                    (sample["depth"][i] - lo) / (hi - lo), 0, 1
                )
                aligned_depths_vis[i, :, :] = torch.clamp(
                    (aligned_depths[i, :, :] - lo) / (hi - lo), 0, 1
                )

            gt_depth_vis = colorize_depth(gt_depth_vis)
            save_RGB(gt_depth_vis, eval_depth_dir, "GT")

            with open(
                os.path.join(eval_depth_dir, f"{self.step}.json"), "w", encoding="utf-8"
            ) as f:
                for i, error in enumerate(errors):
                    dicts = error if isinstance(error, list) else [error]
                    for _, d in enumerate(dicts):
                        record = {
                            **{k: v.item() for k, v in d.items()},
                        }
                        f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

        else:
            aligned_depths_vis = torch.zeros_like(raw_values)
            for i in range(raw_values.shape[0]):
                aligned_depths_vis[i] = (raw_values[i] - raw_values[i].min()) / (
                    raw_values[i].max() - raw_values[i].min()
                )
            aligned_depths = aligned_depths_vis.clone()

        aligned_depths_vis = colorize_depth(aligned_depths_vis)
        save_RGB(aligned_depths_vis, eval_depth_dir, f"{self.step}")
        normal = self.renderer.get_normal_from_depth(scaled_values) * 0.5 + 0.5
        save_RGB(normal, eval_normal_dir, f"{self.step}")

        if self.cfg.logging.save_as_grid:
            stack_images = []
            stack_images.append(sample["org_image"])
            if "depth" in sample:
                stack_images.append(gt_depth_vis)
            stack_images.append(aligned_depths_vis)
            stack_images.append(normal)
            stack_images = torch.stack(stack_images)
            M, B, C, H, W = stack_images.shape
            stack_images = stack_images.permute(1, 0, 2, 3, 4)
            stack_images = stack_images.reshape(B * M, C, H, W)
            grid = make_grid(stack_images, nrow=M, padding=2, pad_value=1)
            save_image(grid, os.path.join(eval_stacked_dir, f"{self.step}.png"))
        if self.cfg.logging.save_model:
            self.save_model(os.path.join(os.path.join(eval_model_dir, f"{self.step}")))
