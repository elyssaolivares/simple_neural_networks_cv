# Can a Simple Neural Network Recognize Images?

**Animal Image Dataset (90 Different Animals)**

| | |
|---|---|
| **Name** | Elyssa Olivares |
| **Section** | BSCS 4B |
| **Date Submitted** | August 28, 2026 |
| **Notebook** | [L2_Olivares.ipynb](https://colab.research.google.com/drive/1ua2vfLg7QcGWDlFBpUbwobUEyDG6Yd-x?usp=sharing) |

---

## 1. Dataset

The Animal Image Dataset (90 Different Animals), curated by Sourav Banerjee on Kaggle, is a multi-class computer vision dataset with images collected from Google Images.

Dataset Link: [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

| Characteristic | Result |
|---|---|
| Classes | 90 |
| Total images | 5,400 (60 per class, perfectly balanced) |
| Format | JPG / JPEG / PNG |
| Dimensions | Varied (resized for training) |
| Color channels | 3 (RGB) |

**Observations:** The classes cover wild mammals, pets, insects, and marine animals. Images vary widely in lighting, background, resolution, and camera angle, so resizing and normalization were needed.

## 2. Data Split

| Set | Images | Share |
|---|---|---|
| Training | 4,320 | 80% |
| Validation | 536 | 9.9% |
| Testing | 544 | 10.1% |

- **Training:** used to update the model weights.
- **Validation:** monitors performance each epoch (early stopping, learning-rate reduction) without updating weights.
- **Testing:** used once at the end for an unbiased estimate of generalization. It is kept separate so the model can't simply memorize the test images.

## 3. Model and Training

| Component | Configuration |
|---|---|
| Input size | 64 × 64 × 3 |
| Architecture | Flatten (12,288 values) → Dense 512 → Dense 256 → Dense 90 |
| Activations | ReLU (hidden), Softmax (output) |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 40 planned, stopped at 26 (EarlyStopping) |
| Batch size | 32 |
| LR schedule | ReduceLROnPlateau, down to 0.0000625 by epoch 24 |

**Flatten layer:** converts the 3D image into a 1D vector with no learnable parameters. Every pixel becomes an independent feature, so the network loses all spatial relationships between neighboring pixels.

**Training curves:**
- Training loss fell from 4.83 to about 3.05.
- Training accuracy rose from about 3% to about 25%.
- Validation accuracy only rose from about 6% to 14%, was noisy, and validation loss stayed above training loss.
- Accuracy was low even on the training set, which points to **underfitting** caused by the Flatten + Dense architecture's inability to capture spatial patterns.

## 4. Experiment: Image Resolution

| Resolution | Input Features | Parameters | Train Acc | Val Acc | Test Acc | Training Time (s) |
|---|---|---|---|---|---|---|
| 32 × 32 | 3,072 | 1,730,906 | 0.2199 | 0.1213 | **0.1434** | 135.5 |
| 64 × 64 | 12,288 | 6,449,498 | 0.1986 | 0.1138 | 0.1268 | 293.4 |
| 128 × 128 | 49,152 | 25,323,866 | 0.2447 | 0.1474 | 0.1048 | 1,749.1 |

**Findings:**
- Higher resolution greatly increases parameters and training time (about 13× longer at 128 × 128 than at 32 × 32).
- Test accuracy did not improve with resolution and was actually lowest at 128 × 128.
- Because Flatten discards spatial structure, extra pixels add cost but little benefit.

## 5. Final Evaluation (64 × 64 model)

| Metric | Result |
|---|---|
| Test loss | 3.8835 |
| Test accuracy | 12.68% |

- **Best class: Jellyfish** (F1 = 0.615). Its distinctive color, translucency, and shape survive flattening.
- **Worst class: Badger** (F1 = 0.000). It overlaps in fur color, texture, and body shape with other mammals, and the model can't use silhouette cues.

## 6. Prediction Analysis

**Correct predictions** were usually animals with distinctive shapes or colors (e.g., rhino, shark, lobster).

| Actual | Predicted | Likely Reason |
|---|---|---|
| Turtle | Shark | Similar color highlights, underwater setting, elongated angle |
| Hippopotamus | Butterfly | Similar width and edges, dark coloring |
| Cow | Sheep | Low resolution blurs edges, similar color |

The network only learns coarse statistics such as color distribution and brightness, so classes with similar colors, habitats, or proportions get confused.

## 7. Conclusion

- **Dataset:** 5,400 balanced images across 90 animal classes, resized and normalized, with a proper train/validation/test split.
- **Performance:** The simple network learned patterns (about 25% training accuracy) but generalized poorly (about 14% validation, 12.68% test). That is still roughly 11 to 12× better than the 1.1% random-guessing baseline.
- **Resolution:** Higher resolution raised parameters and compute cost without improving test accuracy.
- **Main limitation:** The Flatten + Dense design throws away spatial relationships (edges, shapes, textures), and the 90-class task has many look-alike animals.
- **Answer:** A simple neural network can recognize images **only partially**. It beats chance by picking up color and brightness patterns, but it is fundamentally limited for real-world image recognition. Convolutional neural networks (CNNs) close this gap by preserving local pixel neighborhoods through convolution and pooling.
