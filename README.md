# Dimensionality Reduction (PCA · LDA · QDA)

This experiment compares **Principal Component Analysis (PCA)**,  
**Linear Discriminant Analysis (LDA)**, and **Quadratic Discriminant Analysis (QDA)**  
on the **Olivetti Faces dataset** from `sklearn.datasets`.

---

##  Objective

- Understand the mathematical and functional differences between **PCA** and **LDA**.  
- Evaluate their performance in:
  - **Dimensionality reduction**
  - **Image reconstruction**
  - **Classification accuracy**
  - **Computation efficiency**
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

<img width="795" height="364" alt="image" src="https://github.com/user-attachments/assets/65f32fbb-2772-4d30-8c92-134544cd8668" />


### Reconstruction Error vs Components
<img width="562" height="393" alt="image" src="https://github.com/user-attachments/assets/a06f049b-5694-47f5-9acb-f7d686960a7d" />

### Accuracy Comparison
<img width="622" height="393" alt="image" src="https://github.com/user-attachments/assets/c837e874-24ea-40c0-9c99-3af510e77e92" />


### Time Comparison
<img width="613" height="393" alt="image" src="https://github.com/user-attachments/assets/d39a8f1f-7b06-424e-915c-ef41d115dc55" />


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
