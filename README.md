# Anatomically Accurate 3D Vessel Generation via 3-Phase Chebyshev Curve Diffusion

Official implementation of our MICCAI 2026 paper.

## Citation 📄

```bibtex
@inproceedings{mo2026chebyshev,
  title     = {Anatomically Accurate 3D Vessel Generation via 3-Phase Chebyshev Curve Diffusion},
  author    = {Mo, Jihwan and Yoo, Sangbaek and Choi, Jaesoon and Chang, Dong Eui},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  year      = {2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer}
}
```

## Dependencies 📦

```bash
conda create -n vessel python=3.10
conda activate vessel
pip install -r requirements.txt
```

PyTorch must be installed with CUDA matching your GPU. See [pytorch.org](https://pytorch.org/get-started/locally/).

## Dataset 📊

We use the [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT) coronary artery dataset. Place `.vtp` files (containing `Radius` point data) under:

```
data/lca/raw_vtp/     # Left coronary artery
data/rca/raw_vtp/     # Right coronary artery
```

## Usage ⚙️

Replace `lca` with `rca` for right coronary artery (and use `config/curve_config_rca.yaml`).

### 1. Preprocessing
```bash
python scripts/preprocess_tree.py tree --input_dir data/lca/raw_vtp --output_path data/lca.pt
python scripts/preprocess_curve.py --mode all --dataset lca --root .
```

### 2. Stage 1 — Tree VAE
```bash
python train_tree.py --dataset lca --data_path ./data/ --epochs 40000 --batch_size 32
```

### 3. Stage 2 — 3-Phase Curve Diffusion
```bash
python train_diffusion.py --config config/curve_config.yaml --phase 1
python train_diffusion.py --config config/curve_config.yaml --phase 2 --phase1_ckpt <best_magnitude.pt>
python train_diffusion.py --config config/curve_config.yaml --phase 3
```

### 4. Temperature Calibration
```bash
python calibrate_temperature.py \
    --phase1-ckpt <best_magnitude.pt> --phase3-ckpt <best_radius.pt> \
    --npz-dir data/lca/recipe_npz \
    --gt-stats data/lca/statistics/gt_statistics.json \
    --output data/lca/statistics/temperature_config.json
```

### 5. Generation
```bash
python export.py \
    --treevae-ckpt <tree_vae.pth> \
    --phase1-ckpt <best_magnitude.pt> --phase2-ckpt <best_theta.pt> --phase3-ckpt <best_radius.pt> \
    --gt-stats data/lca/statistics/gt_statistics.json \
    --temperature-config data/lca/statistics/temperature_config.json \
    --stats-dir data/lca/statistics \
    --output-dir outputs/lca/export_ply --num-samples 100
```

### 6. Evaluation
```bash
python evaluate.py \
    --gt-dir data/lca/reference_ply \
    --ours-dir outputs/lca/export_ply \
    --eval-mode full
```

For full argument lists, run `python <script>.py --help`.

## Acknowledgments

This work builds upon [PartVessel](https://github.com/CybercatChen/PartVessel) (Chen et al., MICCAI 2025).

## License

MIT — see [LICENSE.txt](LICENSE.txt).
