set -e
for L in "layer1" "layer1 layer2" "layer1 layer2 layer3" "layer2" "layer2 layer3" "layer2 layer3 layer4" "layer3" "layer3 layer4" "layer4"; do
  for N in "joint" "layer_pixel" "layer_pooled"; do
    python run_all.py --root_dir data/MVTec_LOCO_AD_512size \
        --backbone clip_rn101 --layers $L \
        --save_dir runs/clip_rn101_$(echo $N| tr -d ' ') \
        --norm_mod $N
  done
done