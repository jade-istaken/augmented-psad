set -e
for b in "clip_rn101" "resnet101_in1k_v2" "dino_rn50" "random_clip_rn101_s0" "random_clip_rn101_s1" "random_clip_rn101_s2"; do
  python run_all.py --root_dir data/MVTec_LOCO_AD_512size \
      --backbone $b --layers layer1 layer2 layer3 \
      --norm_mode layer_pooled --save_dir runs/backbone_testing/${b}_l123_pooled
done