mkdir -p ./pretrained_models/
wget -O ./pretrained_models/DA3_mono-large.safetensors https://huggingface.co/depth-anything/DA3MONO-LARGE/resolve/main/model.safetensors

git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cp -r Depth-Anything-3/src/depth_anything_3/ ./redepth/model/
rm -rf Depth-Anything-3/