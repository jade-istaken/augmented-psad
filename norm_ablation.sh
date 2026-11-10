for L in "layer3 layer4" "layer1 layer2 layer3"; do
  for N in "layer_pixel" "layer_pooled"; do
    python run_all.py --root_dir data/MVTec_LOCO_AD_512size \
        --backbone clip_rn101 --layers $L \
        --save_dir runs/clip_rn101_$(echo $N| tr -d ' ') \
        --norm_mod $N
  done
done