for L in "layer2" "layer3" "layer4" "layer2 layer3" "layer3 layer4" "layer1 layer2 layer3"; do
  python run_all.py --root_dir data/MVTec_LOCO_AD_512size \
      --backbone clip_rn101 --layers $L \
      --save_dir runs/layer_ablation/clip_rn101_$(echo $L| tr -d ' ')
done