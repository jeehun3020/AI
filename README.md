# Dimensionality Reduction (PCA · LDA · QDA)

This experiment compares **Principal Component Analysis (PCA)**,  
**Linear Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)**  
on the **Olivetti Faces dataset** from `sklearn.datasets`.

---

##  Objective

- Understand the mathematical and functional differences between **PCA** and **LDA**.  
- Evaluate their performance in:
  - 🧩 **Dimensionality reduction**
  - 🎨 **Image reconstruction**
  - 🎯 **Classification accuracy**
  - ⚙️ **Computation efficiency**
- Additionally, test **QDA** to analyze the effect of *covariance sharing* in discriminant models.

---

##  Dataset

- **Olivetti Faces Dataset**  
  - 400 grayscale facial images (64×64 pixels)  
  - 40 individuals × 10 images each  
  - Each image represented as a 4096-dimensional vector  
- Split: `Train 80% / Test 20% (stratified)`

---

## Methods

| Model | Type | Description |
|:--|:--|:--|
| **PCA** | Unsupervised | Projects data to directions of maximum variance |
| **LDA** | Supervised | Projects data to maximize class separability |
| **QDA** | Supervised | Independent covariance per class (no sharing) |

- Classifier: **Linear SVM**
- Evaluation metrics:
  - Accuracy
  - Reconstruction Error (PCA only)
  - Computation Time

---

## Results

| Model | Accuracy | Time (s) | Components |
|:--|--:|--:|:--|
| **PCA** | 0.975 | 1.01 | 100 |
| **LDA** | 1.000 | 0.46 | 39 |
| **QDA** | 0.012 | 0.04 | class-specific |

---

##  Visualizations

### PCA Reconstruction
Shows how PCA reconstructs test faces using top principal components.

| Original | Reconstructed |
|:--:|:--:|
| ![orig1](../assets/pca_recon1.png) | ![recon1](../assets/pca_recon2.png) |

### Reconstruction Error vs Components
![Reconstruction Error](../assets/pca_mse_curve.png)

### PCA vs LDA Feature Space
![Feature Space](../assets/pca_lda_features.png)

### Accuracy & Time Comparison
| Accuracy | Computation Time |
|:--:|:--:|
| ![Acc](../assets/accuracy_bar.png) | ![Time](../assets/time_bar.png) |

---

## Analysis & Key Findings

- **PCA**
  - High reconstruction fidelity  
  - Slightly slower due to SVD computation  
  - Unsupervised, so clusters overlap  

- **LDA**
  - Perfect classification (100%)  
  - Forms clear class clusters  
  - Supervised and computationally efficient  

- **QDA**
  - Drastically lower accuracy due to non–full-rank covariance  
  - Confirms that *covariance sharing (LDA)* stabilizes performance in high-dimensional spaces  

---

## Conclusion

- **PCA** preserves data structure → suitable for reconstruction or compression.  
- **LDA** enhances class separability → ideal for supervised recognition.  
- **QDA** reveals the instability of independent covariance estimation in limited samples.  
- This experiment confirms that **covariance sharing is critical for robust classification**  
  in high-dimensional facial datasets.

---

### Author
> Jihoon Jeong (정지훈)  
> Department of AI Computer Engineering, Kyonggi University  
> [GitHub: jeehun3020](https://github.com/jeehun3020)
