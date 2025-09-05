# Leveraging Generative AI to Generate and Detect Autistic Facial Expressions  

## 🔹 Overview  
This project explores how **Generative Adversarial Networks (GANs)** and **Deep Learning classifiers** can be used to generate and detect autistic facial expressions. The goal is to both expand underrepresented datasets (especially Middle Eastern children) and improve early detection through automated image classification.  

---

## 📌 Problem Statement  
Autism diagnosis often relies on behavioral assessments that can be subjective, time-consuming, and vary across cultures.  
This project investigates whether AI can:  
1. **Generate synthetic facial images** representing autistic vs. non-autistic traits to enrich limited datasets.  
2. **Classify faces** into autistic and non-autistic categories using transfer learning.  

---

## 📊 Dataset  
- **Total size:** ~1,600+ images per class (`autistic`, `non_autistic`).  
- **Sources:** Public datasets + curated/augmented samples.  
- **Challenges:** Limited representation of Middle Eastern children (~20 images initially).  


**Preprocessing:**  
- Resized to **512×512** for GAN training.  
- Augmented with flips, rotations, and slight color adjustments.  
- Balanced classes before classifier training.  

---

## ⚙️ Methods  

### 1. Generative Modeling  
- **Model:** [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada)  
- **Training setup:**  
  - Image size: 512×512  
  - Fine-tuned for 1.5 weeks on custom dataset  
  - Mixed dataset (400 images per class, including augmentations + Middle Eastern samples)  
- **Results:** Achieved **FID = 15.2**, with realistic synthetic faces.  

### 2. Classification  
- **Model:** EfficientNetB3 & EfficientNetB4 (transfer learning)  
- **Training:**  
  - Custom train/validation/test splits via Pandas DataFrames  
  - Standardized input features  
  - Loss: Categorical cross-entropy  
  - Optimizer: Adam  
- **Results:** Best model (EfficientNetB4) reached **92% accuracy** on unseen test data.  

---

## 📈 Results  

### GAN Outputs  
*(Synthetic examples generated with StyleGAN2-ADA)*  
![GAN Outputs](images/generated_faces.png)  

### Training Curves  
*(Accuracy and loss across epochs)*  
![Training Curves](images/training_curve.png)  

### Classifier Performance  
- **Validation Accuracy:** 90%  
- **Test Accuracy:** 92%  
- **Observation:** Model captured subtle facial cues but struggled with underrepresented ethnic groups.  

---

## 🔑 Key Findings  
- GAN-generated images improved dataset diversity, though ethnic imbalance remained a challenge.  
- EfficientNet models were able to detect autistic vs. non-autistic features with high accuracy.  
- Balancing and standardizing the dataset was critical to stability and fairness.  

---

## 🚀 How to Run  

1. Clone repo  
```bash
git clone https://github.com/yourusername/Autism-Facial-Expression-Detection.git
cd Autism-Facial-Expression-Detection

