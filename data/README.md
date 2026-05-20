# Data Directory

## Structure

```
data/
├── lca/                          # Left Coronary Artery
│   ├── raw_vtp/                  # Raw .vtp centerline files
│   ├── recipe_npz/               # Preprocessed Chebyshev coefficients (.npz)
│   ├── reference_ply/            # Ground truth point clouds for evaluation (.ply)
│   └── statistics/               # Dataset statistics (gt_statistics.json)
└── rca/                          # Right Coronary Artery
    ├── raw_vtp/                  # Raw .vtp centerline files
    ├── recipe_npz/               # Preprocessed Chebyshev coefficients (.npz)
    ├── reference_ply/            # Ground truth point clouds for evaluation (.ply)
    └── statistics/               # Dataset statistics (gt_statistics.json)
```

## Data Preparation

1. Place your coronary artery `.vtp` files in `{lca,rca}/raw_vtp/`
2. Run preprocessing scripts (see main README.md)
3. Reference PLY files are generated during the preprocessing step for evaluation
