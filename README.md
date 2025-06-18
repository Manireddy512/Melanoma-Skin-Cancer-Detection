# Melanoma Skin Cancer Detection using Deep Learning (CNN)

A deep learning-based classification system built from scratch using Convolutional Neural Networks (CNN) to detect melanoma and other types of skin lesions from dermoscopic images. This project addresses early skin cancer diagnosis using image analysis and AI techniques, without relying on pretrained models.

---

## Project Objective

- Develop a custom CNN model (no transfer learning) for multi-class classification of 9 types of skin lesions.
- Improve diagnostic reliability and interpretability through end-to-end learning on dermoscopic images from the ISIC dataset.

---

## Technologies Used

- **Python**  
- **TensorFlow** / **Keras**  
- **Augmentor** (for image augmentation)  
- **Google Colab** (training environment)  
- **Matplotlib**, **Seaborn**, **NumPy**, **Pandas**  

---

## Dataset

- **Source**: ISIC (International Skin Imaging Collaboration) Archive  

![alt text](https://github.com/Manireddy512/Melanoma-Skin-Cancer-Detection/blob/8f2ba0c540bd157774835272f1c2745771368609/Images/DatasetData.png)

- **Classes**: 9 types of skin diseases (malignant and benign)  

![alt text](https://github.com/Manireddy512/Melanoma-Skin-Cancer-Detection/blob/8f2ba0c540bd157774835272f1c2745771368609/Images/DataSetClasses.png)


- **Size**: 2,357 original images → Balanced using augmentation  

![alt text](https://github.com/Manireddy512/Melanoma-Skin-Cancer-Detection/blob/8f2ba0c540bd157774835272f1c2745771368609/Images/DatasetData.png)
- **Input Shape**: 180 x 180 x 3 RGB  
- **Train/Validation/Test Split**: 80/20 split used with augmentation on training data  

---

## CNN Architecture Highlights

- Rescaling layer to normalize inputs  
- 3 Convolutional + MaxPooling layers  
- Dropout layers for regularization  
- Dense + ReLU + Softmax for final multi-class prediction  
- Trained using Adam optimizer with categorical cross-entropy loss 

![alt text](https://github.com/Manireddy512/Melanoma-Skin-Cancer-Detection/blob/8f2ba0c540bd157774835272f1c2745771368609/Images/ModelArc.png)

---

## Evaluation Metrics

- **Accuracy**: ~84.26% on validation data  
- **Precision / Recall / F1-score**: Calculated per class  
- **Confusion Matrix** and **Training Curves** for performance visualization  

![alt text](https://github.com/Manireddy512/Melanoma-Skin-Cancer-Detection/blob/8f2ba0c540bd157774835272f1c2745771368609/Images/Evaluation%20Metrics.png)

---

## Key Features

- Custom-built CNN (no transfer learning used)  
- Balanced data using **Augmentor** for robust generalization  
- Focus on **interpretability and control**, suitable for clinical integration  
- Supports **multi-class classification** across 9 lesion types  

---

## Results Summary

- **Best Validation Accuracy**: 84.26%  
- **Model performs best on melanoma, nevus, vascular lesions**  
- Demonstrates potential as a diagnostic aid for dermatologists  

---

## How to Run

1. Clone the repo and open `melanoma_skin_cancer_detection.ipynb` in Google Colab
2. Mount your Google Drive and load the dataset
3. Run all cells to:
   - Preprocess and augment data
   - Build and train the CNN
   - Evaluate and visualize model performance

---

## Dataset Access

-  [ISIC Archive (Dataset)](https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view)  


---

## License

This project is for academic and educational purposes only. Please contact if you'd like to use or expand this work for commercial or research deployment.

---
## References

- [Melanoma Skin Cancer – American Cancer Society](https://www.cancer.org/cancer/melanoma-skin-cancer/about/what-is-melanoma.html)  
- [Introduction to CNN – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/)  
- [Image Classification using CNN – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/)  
- [Efficient CNN Architecture – Towards Data Science](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7)
