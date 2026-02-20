import torch
import torch.nn as nn
import torch.nn.functional as F

dtype_mapping = {
    "torch.float64": torch.float64,
    "torch.float32": torch.float32,
    "torch.int64": torch.int64,
}


class ShadedDepthRenderer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device
        self.dtype = dtype_mapping[self.cfg.dtype]

        self.view_d = torch.tensor(
            self.cfg.view_d, device=self.device, dtype=self.dtype
        )
        self.light_d_min = self.cfg.light_d_min
        self.light_d_max = self.cfg.light_d_max

        self.shininess_exponent_min = self.cfg.shininess_exp_min
        self.shininess_exponent_max = self.cfg.shininess_exp_max

    def forward(self, depth, image):
        normal = self.get_normal_from_depth(depth)
        rand_light_d = torch.cat(
            [
                (
                    (
                        torch.empty(
                            normal.shape[0], 2, device=self.device, dtype=self.dtype
                        ).uniform_(self.light_d_min, self.light_d_max)
                    )
                ),
                torch.ones(normal.shape[0], 1).to(self.device),
            ],
            dim=1,
        )
        rand_light_d = rand_light_d / torch.norm(rand_light_d, p=2, dim=1, keepdim=True)
        halfway_v = rand_light_d + self.view_d
        halfway_v = halfway_v / (
            torch.linalg.norm(halfway_v, dim=1, keepdim=True) + 1e-8
        )

        diffuse_shading = (normal * rand_light_d.view(-1, 3, 1, 1)).sum(1, keepdim=True)
        diffuse_shading = torch.clamp(diffuse_shading, min=0.0)

        specular_shading = (normal * halfway_v.view(-1, 3, 1, 1)).sum(1, keepdim=True)
        shininess = (
            2
            ** torch.randint(
                self.shininess_exponent_min, self.shininess_exponent_max, (1,)
            ).item()
        )
        specular_shading = torch.clamp(specular_shading, min=0.0) ** shininess

        diffuse_spec_coeff = torch.rand(2, device=self.device, dtype=self.dtype)
        diffuse_spec_coeff = diffuse_spec_coeff / diffuse_spec_coeff.norm(p=1)

        rendered_image = (
            diffuse_spec_coeff[0] * diffuse_shading * image
            + diffuse_spec_coeff[1] * specular_shading
        )
        rendered_image = torch.clamp(rendered_image, min=1e-3, max=1.0)

        return rendered_image

    def get_normal_from_depth(self, depth):

        B, H, W = depth.shape

        aspect = W / H

        u = torch.linspace(-aspect, aspect, W, dtype=depth.dtype, device=depth.device)
        v = torch.linspace(1, -1, H, dtype=depth.dtype, device=depth.device)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        vv = vv.unsqueeze(0).repeat(B, 1, 1)
        uu = uu.unsqueeze(0).repeat(B, 1, 1)

        z = depth

        valid = torch.isfinite(z) & (z > 0)

        points = torch.stack((uu, vv, z), dim=-1)

        gy, gx = torch.gradient(points, dim=(1, 2))

        n = torch.cross(gy, gx, dim=-1)

        n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)

        n = torch.where(
            valid[..., None],
            n,
            torch.tensor([0.0, 0.0, -1.0], dtype=depth.dtype, device=depth.device),
        )

        return n.permute(0, 3, 1, 2)
