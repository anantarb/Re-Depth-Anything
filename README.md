# Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting

<p align="center"><img src="./assets/teaser.gif" alt="animated"/></p>

Re-Depth refines the prediction of the monocular depth estimators via novel, self-supervised re-lighting method.

This is the official repository that contains source code for the arXiv paper [Re-Depth Anything](https://arxiv.org/pdf/2512.17908).

If you find Re-Depth Anything useful for your work please cite:
```
@article{bhattarai2025redepth,
      title={Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting},
      author={Bhattarai, Ananta R. and Rhodin, Helge},
      journal={ArXiv},
      year={2025}
}
```

## News

* **15.02.2026**: Code for Re-Depth Anything is released. :fire:


## Prepare Environment

1. Clone the repository, then run the scripts to configure your environment and download required assets:

```
git clone https://github.com/anantarb/Re-Depth-Anything.git
cd Re-Depth-Anything/
source ./scripts/install_deps.sh
sh ./scripts/prepare_DAv2_small.sh
```
The first script creates the `redepth` conda environment and installs all necessary packages. The second script downloads the pre-trained models and supporting modules, and places them into the appropriate folders.

This project was developed and tested on **Ubuntu 24.04**, **CUDA 12.8**, and **Python 3.12**. If you are using a different OS, CUDA version, or Python version, you may need to make additional adjustments to get the environment working. 

If you run into dependency issues, you can refer to `requirements_dev.txt`, which lists the exact package versions used during development.

## Running

### Preparing the config

We provide an example config at [`redepth/config/DAv2small_example.yaml`](redepth/config/DAv2small_example.yaml). To run Re-Depth on your own image, make the following changes:

- **`image_path`**: Set this to the path of your input image.
- **`depth_path`** *(optional)*: Set this to the path of your ground-truth (GT) depth map, if available. When provided, the output will be compared against GT **both qualitatively and quantitatively**, and the results will be saved in the logging directory.
- **`depth_min_value` / `depth_max_value`** *(optional)*: Adjust these if you want to evaluate only GT depth values within a specific range. If not set, the defaults are:
  - `depth_min_value = 1e-3`
  - `depth_max_value = 1e4`
- **`mask`** *(optional)*: Set this if you have an image mask. Note that passing `mask` does **not** affect the optimization process.
- **`depth_mask`** *(optional)*: Set this if you want evaluation against GT to be limited to the regions defined by `depth_mask`. If `depth_mask` is not provided, it is automatically derived from the GT depth values using `depth_min_value` and `depth_max_value`.
- **`depth_to_meters`** *(optional)*: Set this scaling factor to convert the GT depth units to meters for evaluation.
- **`save_dir`**: Set this to the directory where you want outputs to be saved.

For details on how inputs are loaded and processed, see [`redepth/dataset/DAv2_dataset.py`](redepth/dataset/DAv2_dataset.py).

The items above are the minimum changes needed to run Re-Depth on your own image. For best results, you may want to tune additional hyperparameters such as `lr`, `scale`, `smoothness_weight`, and `max_steps`.   

### Running Re-Depth Anything

After configuring your setup, run the optimization script:
```bash
python redepth/scripts/run_redepth_DAv2.py --config=PATH_TO_CONFIG
```

#### Output directory + naming conventions

Outputs are saved to `save_dir` every `save_interval` (as specified in your config). The value `{X}` (see below) is computed as:
`X = (current_ensemble_size * max_step + current_step)`

where:
- `current_ensemble_size ∈ [0, ensemble_size - 1]`
- `X = 0` always refers to the **initial model output** (no optimization applied).

##### Depth outputs (ensembled)

- `0_{X}.png`: Depth visualized using the `cv2.COLORMAP_INFERNO` colormap.
- `{X}_raw.pt`: Raw model output with no transformations applied.
- `{X}_scaled.pt`: Scaled orthographic model output (Eq. 5 in the main paper).
- `{X}.json`: Quantitative evaluation of the output against GT depth.
- `GT.pt`: Resized ground truth depth.

##### Normal outputs (computed from ensembled depths)

- `0_{X}.png`: Normals computed from the scaled orthographic depth using Eq. 3 in the main paper.

##### Model outputs (not ensembled)

- `{X}.pt`
  - Checkpoints are saved from the **current step** within each ensemble run (the models themselves are **not** ensembled).

#### Stacked outputs

- `{X}.png`: Input image, GT depth (if available), output depth (ensembled), and normals from the scaled orthographic output depth (ensembled) stacked together for visualization.

## Applying Re-Depth to Other Models

- **Re-Depth Anything** can be used with other backbone models (e.g., **Depth-Anything-V2-Giant**, **DA3MONO-LARGE**). To do this, create a model-specific coach/trainer class (see `redepth/coach/DAv2_coach.py`) that **inherits** from the base coach at `redepth/coach/base_coach.py`.
- The base coach expects the derived class to provide the required **attributes** (e.g., `Dataset`, `Model`, `Optimizer`) and to implement the required **methods** (listed under `@abstractmethod`) so the optimization can be launched.
- Keep in mind that different base models output **depth/disparity in different value ranges**. As a result, the configuration values for `scale` and `ms` may need to be **significantly different** across models. Since Re-Depth relies on the **scaled depth** computed from these parameters, getting good estimates for `scale` and `ms` is important for best performance.
- Other hyperparameters can also strongly affect results, including `smoothness_weight`, `embeddings_lr`, `dpt_lr`, and `scale_lr`.
- We plan to expand the codebase to support more base models in the future.


## Acknowledgements

Our work builds on top of amazing open-source networks and codebases. 
We thank the authors for providing them.

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): a SOTA monocular depth estimator.
- [threestudio](https://github.com/threestudio-project/threestudio): a unified framework for 3D content creation from text prompts, single images, and few-shot images, by lifting 2D text-to-image generation models.
- [Hugging Face](https://github.com/huggingface): a platform that provides libraries for many machine learning tasks like text generation, image generation, and many more. 

