# Dataset Preparation Notes

## SAM-DD (RGB) hierarchy

The SAM-DD videos are exported as images using the folder layout
`Tester*/<class>/<view>/img_xxx.jpg`. Because the semantic class is not the direct
parent directory, the default ImageFolder loader treats each `Tester*` directory as a
distinct label, which quickly leads to overfitting and prevents proper shuffling.

Use the helper script to flatten the hierarchy into a CSV with explicit labels:

```bash
python -m DA.tools.build_samdd_labels \
  --data-dir Dataset/SAM-DD(RGB) \
  --output Dataset/samdd_labels.csv \
  --subject-glob "Tester*"
```

This writes columns `subject`, `classname`, `view`, `img`, and `filepath`. The new
`filepath` column lets `CSVImageDataset` resolve each sample relative to the dataset
root, so you can keep the original directory structure.

### Training with shuffled/global labels

```bash
python -m DA.train \
  --data-dir Dataset/SAM-DD(RGB) \
  --labels-csv Dataset/samdd_labels.csv \
  --group-column subject \
  --batch-size 64 \
  --epochs 20 \
  --model-name resnet50
```

Passing `--group-column subject` keeps entire Tester identities in the same split to
reduce leakage. Drop the flag if you prefer a fully shuffled split.

### Evaluation

Use the same CSV to ensure the checkpoint sees the correct labels:

```bash
python -m DA.evaluate \
  --checkpoint runs/my_exp/best.ckpt \
  --data-dir Dataset/SAM-DD(RGB) \
  --labels-csv Dataset/samdd_labels.csv
```

