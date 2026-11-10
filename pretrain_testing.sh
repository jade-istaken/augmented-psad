set -e
for s in 0 1 2; do
  python run_all.py --root_dir data/MVTec_LOCO_AD_512size \
      --backbone random_clip_rn101_s$s --layers layer1 layer2 layer3 \
      --norm_mode layer_pooled --save_dir runs/pretrain_testing/random_clip_rn101_s${s}_l123_pooled
done