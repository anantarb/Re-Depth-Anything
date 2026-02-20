mkdir -p ./pretrained_models/
wget -P ./pretrained_models/ https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cp -r Depth-Anything-V2/depth_anything_v2/ ./redepth/model/
rm -rf Depth-Anything-V2/