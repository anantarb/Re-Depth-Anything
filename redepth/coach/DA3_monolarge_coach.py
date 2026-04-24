import torch.nn.functional as F
import torch
import torch.nn as nn
import gc
from safetensors.torch import load_file
from redepth.coach.base_coach import Trainer

from redepth.model.depth_anything_3.model.dinov2.dinov2 import DinoV2
from redepth.model.depth_anything_3.model.dpt import DPT
from redepth.dataset.DAv2_dataset import ReDepthDAv2Dataset

dtype_mapping = {
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
    "torch.int64": torch.int64,
}


class Model(nn.Module):

    def __init__(self, cfg, train_dataset):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = dtype_mapping[self.cfg.dtype]
        self.train_dataset = train_dataset
        self._initialize_model()
        self._initialize_optimizers()

    def forward(self, x):

        x = tuple(zip(self.embeddings, self.embedding_extras))

        raw_depth = self.depth_head(x, self.height, self.width, patch_start_idx=0)[
            "depth"
        ]
        scale = self.scale + torch.tanh(self.scale_offset)
        depth = raw_depth * scale
        return raw_depth.squeeze(1), depth.squeeze(1)

    def _initialize_model(self):

        self.pretrained = (
            DinoV2(**self.cfg.depth_model.net).to(self.device).to(self.dtype)
        )

        self.depth_head = (
            DPT(**self.cfg.depth_model.head).to(self.device).to(self.dtype)
        )
        self._load_checkpoint(self.cfg.depth_model.checkpoint_path)
        input_image = (
            self.train_dataset.__getitem__(idx=0)["input_image"]
            .to(self.device)
            .to(self.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        with torch.no_grad():
            embedding_features, aux_features = self._infer_embeddings(input_image)
        del self.pretrained

        gc.collect()
        torch.cuda.empty_cache()

        self.embeddings = torch.nn.ParameterList(
            [torch.nn.Parameter(f[0].clone().detach()) for f in embedding_features]
        )

        self.embedding_extras = torch.stack(
            [f[1].clone().detach() for f in embedding_features]
        )

        self.scale = self.cfg.depth_model.scale
        self.scale_offset = nn.Parameter(
            torch.tensor(0, dtype=self.dtype, device=self.device)
        )

    def _initialize_optimizers(self):
        param_groups = []
        param_groups.append(
            {
                "params": self.embeddings.parameters(),
                "lr": self.cfg.training.embeddings_lr,
            }
        )
        param_groups.append(
            {"params": self.depth_head.parameters(), "lr": self.cfg.training.dpt_lr}
        )
        param_groups.append(
            {
                "params": self.scale_offset,
                "lr": self.cfg.training.scale_lr,
            }
        )

        self.optimizer = torch.optim.AdamW(param_groups)

    def _infer_embeddings(self, image):
        self.height, self.width = image.shape[-2], image.shape[-1]
        self.patch_h, self.patch_w = image.shape[-2] // 14, image.shape[-1] // 14

        features = self.pretrained.forward(
            image,
            cam_token=None,
            export_feat_layers=self.cfg.depth_model.net.out_layers,
            ref_view_strategy="saddle_balanced",
        )
        return features

    def _load_checkpoint(self, ckpt_path):
        ckpt = load_file(ckpt_path)
        self.pretrained.load_state_dict(
            {k[15:]: v for k, v in ckpt.items() if "pretrained" in k}, strict=True
        )
        self.depth_head.load_state_dict(
            {k[11:]: v for k, v in ckpt.items() if "head" in k}, strict=True
        )

    def get_optimizers(self):
        return self.optimizer


class DA3Trainer(Trainer):

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = dtype_mapping[self.cfg.dtype]

        self.train_dataset = ReDepthDAv2Dataset(self.cfg.data)
        self.test_dataset = ReDepthDAv2Dataset(self.cfg.data)
        self.model = Model(self.cfg, self.train_dataset)
        self.optimizer = self.model.get_optimizers()
        super().__init__(cfg)

    def reset_depth_model(self):
        self.model = Model(self.cfg, self.train_dataset)
        self.optimizer = self.model.get_optimizers()

    def save_model(self, filename):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"{filename}.pt",
        )
