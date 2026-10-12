# Chest X-Ray Classification Using CNNs

End-to-end binary image-classification experiments for **NORMAL** versus **PNEUMONIA**, implemented in `CNN_PIPELINE.ipynb` using TensorFlow/Keras. The primary model is a custom CNN, compared with frozen and fine-tuned VGG16 and ResNet50 models.

## Dataset and preprocessing

The notebook downloads the Kaggle dataset identified by `paultimothymooney/chest-xray-pneumonia` through `kagglehub`.

1. Combine images from the original `train`, `val`, and `test` folders into one dataframe.
2. Inspect class balance, image dimensions, and image modes.
3. Remove unreadable images and exact duplicates using SHA-256 hashes of RGB pixel data and image size. Exclude identical images with conflicting labels.
4. Convert images to RGB, resize to **224 × 224** using bilinear interpolation, and scale pixels to **[0, 1]**.
5. Create a new stratified **80% training / 10% validation / 10% test** split with seed **42**.
6. Compute balanced training class weights and verify that rows and exact image hashes do not overlap across splits.

| Label | Encoded value |
|---|---:|
| NORMAL | 0 |
| PNEUMONIA | 1 |

The model uses three-channel input, even when the original X-ray is grayscale. Training augmentation includes rotation, zoom, translation, and brightness changes, with light and strong settings. Validation and test images are not augmented.

## Models

The custom CNN consists of:

1. Input and training augmentation, with optional rescaling to [-1, 1].
2. Three convolution/max-pooling blocks with 32, 64, and 128 filters; 3 × 3 convolutions, ReLU, and 2 × 2 pooling.
3. Flatten, a 64-unit ReLU dense layer with L2 regularization, and dropout.
4. One sigmoid output representing the pneumonia score.

Training uses Adam, binary cross-entropy, accuracy, and ROC-AUC. Checkpointing and early stopping monitor **validation AUC**.

VGG16 and ResNet50 use ImageNet weights, model-specific Caffe-style preprocessing, global average pooling, and a binary classification head. Each baseline first trains with its backbone frozen, then fine-tunes its final convolutional block at one-tenth of the initial learning rate. Batch-normalization layers stay frozen.

## Setup

Run the notebook in Google Colab with a GPU runtime. Install missing dependencies in a notebook cell:

```python
%pip install tensorflow numpy pandas matplotlib pillow scikit-learn kagglehub
```

Dataset and pretrained-weight downloads require internet access. Authenticate with Kaggle if prompted. The notebook does not pin dependency versions.

The CNN stores image arrays in RAM, so allow several GB of available memory in addition to training overhead.

## How to run

1. Open `CNN_PIPELINE.ipynb` in Colab and select a GPU runtime.
2. Install dependencies and run the cells from top to bottom.
3. Check that `DATASET` points to the downloaded `chest_xray` directory.
4. Mount Google Drive when prompted. The default output directory is `/content/drive/MyDrive/chest_xray/tuning`.
5. Run the one-epoch `code_check` before launching all tuning trials.
6. Complete tuning, inspect the selected hyperparameters, and run the training-data-volume study.
7. Train the comparison models, then run evaluation and error analysis.

For local Jupyter, remove the Colab import and Drive mount in the output-configuration cell and replace the path with:

```python
OUTPUT = Path("./chest_xray/tuning")
OUTPUT.mkdir(parents=True, exist_ok=True)
```

## Experiments

The notebook samples **30 configurations from 256 possible combinations** with `ParameterSampler(random_state=42)`. Although some headings say “grid search,” the implemented search is randomized.

| Hyperparameter | Values searched |
|---|---|
| Batch size | 32, 64 |
| Learning rate | 0.001, 0.0001 |
| Maximum epochs | 15, 30 |
| Dropout | 0.3, 0.5 |
| Early-stopping patience | 3, 5 |
| L2 coefficient | 0.0001, 0.001 |
| Normalization | `zero_one`, `two_x_minus_one` |
| Augmentation | `light`, `strong` |

The selected configuration has the highest validation AUC. The data-volume study retrains that configuration on class-balanced proportions of **25%, 50%, 75%, and 100%** of the training data, using fixed validation data and the selected epoch limit and patience.

## Evaluation and outputs

All five model variants are evaluated on the same test split. Classification uses a threshold of **0.5**. Outputs include accuracy, per-class and macro precision/recall/F1, ROC-AUC, confusion matrices, ROC curves, predictions, and custom-CNN error examples.

| Output | Contents |
|---|---|
| `clean_dataset.csv`, `cleaning_summary.csv` | Cleaned image metadata and cleaning counts |
| `corrupted_images.csv`, `duplicate_copies.csv` | Removed-image records |
| `split_indices.npz`, `split_counts.csv` | Reproducible split indices and class counts |
| `search_manifest.json`, `grid_results.csv` | Search configuration and completed trials |
| `hyperparameter_table.csv`, `custom_cnn_layer_table.csv` | Selected settings and architecture |
| `*.keras`, `*_history.json`, `*_curves.png` | Checkpoints, training histories, and saved curves |
| `data_volume_results.csv` | Training-size experiment results |
| `model_paths.json`, `comparison_training_protocol.csv` | Comparison checkpoints and training budgets |
| `model_comparison.csv` | Test metrics across model variants |
| `*_predictions.csv`, `*_report.csv`, `*_confusion_matrix.csv`, `*_evaluation.png` | Per-model evaluation artifacts |
| `sample_predictions.png`, `misclassified_images.png`, `error_review.csv` | Qualitative predictions and manual error-review template |

## Reproducibility and implementation notes

- Tuning resumes completed run IDs from `grid_results.csv`. A manifest prevents mixing changed search settings or splits with previous results; use a fresh `OUTPUT` directory when the manifest differs.
- Two consecutive search-loop cells contain the same code. The second skips trials already recorded by the first.
- The split is image-level, not patient-grouped. Exact-duplicate checks do not establish patient-level independence.
- The new test split differs from the dataset's original test split, so comparisons must use the same split protocol.
- Fine-tuned baselines include their preceding frozen-stage training; consult `comparison_training_protocol.csv` when comparing training budgets.
- In a fresh Python session, define `CaffePreprocessing` before loading baseline checkpoints.
- This README documents the supplied code. Training was not rerun to produce new performance claims.
