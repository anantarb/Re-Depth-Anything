import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import cv2
import gc
from PIL import Image
from torch.utils.data import Dataset
from redepth.dataset.util.DAv2transform import Resize, NormalizeImage, PrepareForNet
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class ReDepthDAv2Dataset(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg
        org_image = cv2.imread(cfg.image_path)
        org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB) / 255.0

        sample = {
            "org_image": org_image,
            "input_image": org_image.copy(),
        }

        if self.cfg.depth_path is not None:
            if self.cfg.depth_to_meters is not None:
                depth = (
                    cv2.imread(cfg.depth_path, cv2.IMREAD_UNCHANGED).astype("float32")
                    / self.cfg.depth_to_meters
                )
            else:
                depth = cv2.imread(cfg.depth_path, cv2.IMREAD_UNCHANGED).astype(
                    "float32"
                )
            sample["depth"] = depth

        if self.cfg.depth_mask is not None:
            depth_mask = cv2.imread(self.cfg.depth_mask, cv2.IMREAD_GRAYSCALE) / 255.0
            sample["depth_mask"] = depth_mask

        elif self.cfg.depth_path is not None and self.cfg.depth_mask is None:
            depth_mask = np.ones_like(depth)
            if self.cfg.depth_min_value is not None:
                depth_mask = np.logical_and(
                    depth_mask, depth > self.cfg.depth_min_value
                )
            else:
                depth_mask = np.logical_and(depth_mask, depth > 1e-3)
            if self.cfg.depth_max_value is not None:
                depth_mask = np.logical_and(
                    depth_mask, depth < self.cfg.depth_max_value
                )
            else:
                depth_mask = np.logical_and(depth_mask, depth < 1e4)
            sample["depth_mask"] = depth_mask

        if self.cfg.mask is not None:
            object_mask = cv2.imread(self.cfg.mask, cv2.IMREAD_GRAYSCALE) / 255.0
            sample["mask"] = object_mask

        self.transform = Compose(
            [
                Resize(
                    width=self.cfg.width,
                    height=self.cfg.height,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method=self.cfg.resize_method,
                    image_interpolation_method=cv2.INTER_LINEAR,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        sample = self.transform(sample)

        self.sample = {}

        for key, value in sample.items():
            self.sample[key] = torch.from_numpy(value).clone().detach()

        self.sample["prompt"] = self._get_image_prompt(cfg.image_path)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample

    def _get_image_prompt(self, image_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blip2_processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b", cache_dir="./.cache"
        )

        blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            load_in_8bit=True,
            device_map=device,
            dtype=torch.float16,
            cache_dir="./.cache",
        )

        blip2_inputs = blip2_processor(
            images=Image.open(image_path), return_tensors="pt"
        ).to(device, torch.float16)

        generated_ids = blip2_model.generate(**blip2_inputs)
        prompt = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()

        del blip2_processor
        del blip2_model
        gc.collect()
        torch.cuda.empty_cache()

        return prompt
