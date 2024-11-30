# PolytechNice2023: Deep Learning for Insect Classification
This project was developed for the PolytechNice2023 Deep Learning for Media competition, for computer vision and machine learning course. The goal is to classify insect species using the IP102 dataset with over 75,222 images of 102 insect species.
## Overview
- **Objective**: Build a robust classification model for 102 insect species.
- **Dataset**: IP102 Dataset, with highly imbalanced classes.
- **Techniques**: Deep learning methods using PyTorch and Keras, experimenting with multiple architectures and optimization strategies.
- **Competition**: Hosted on Kaggle.

## Progress Highlights

### Framework Selection
We started by exploring both **Keras** and **PyTorch**, ultimately selecting PyTorch due to better performance and flexibility in training.


### Challenges and Solutions
- **GPU Activation**: Initially struggled with GPU setup, but enabling it allowed us to increase training epochs from 5 to 10, reducing model training time from 5 hours to under 2 hours.
- **Imbalanced Dataset**: Used data augmentation (e.g., brightness and rotation) to balance the dataset and improve model generalization.

### Model Development
Our approach evolved through three stages:

#### **VGG16 Pretrained Model (TensorFlow)**:
- Initial attempt with VGG16 achieved only ~1.2% accuracy.
- Modified layers and optimizer (RMSprop) but faced poor results due to architecture limitations.

#### **ResNet50 Pretrained Model (TensorFlow)**:
- Transitioned to ResNet50, achieving higher accuracy and F1 scores.
- Identified data loading issues, which led to inconsistent predictions despite promising training metrics.

#### **ResNet50 Pretrained Model (PyTorch)**:
- Final implementation with PyTorch and ResNet50 yielded the best F1 score.
- Enhanced training with batch size adjustments (128) and expanded data augmentation.

### Key Results
- Significant accuracy improvement over the naive baseline.
- Efficient model training using optimized architecture and techniques.
- Solid foundation for further exploration and improvement.

### Future Improvements
- Explore alternative pretrained models.
- Use callbacks like early stopping and learning rate scheduling.
- Experiment with different loss functions, optimizers, and more epochs.
- Implement advanced data augmentation techniques (e.g., zoom).
- Fine-tune more layers of the pretrained model.
- Employ model ensembling to boost performance.

## Conclusion
This project demonstrates the iterative nature of deep learning, requiring continuous experimentation to optimize performance. A thorough understanding of the dataset and careful selection of data augmentation and model architecture were crucial in achieving competitive results.


## Citation
Kevin Mottin and Oshillou. *PolytechNice2023 DL 4 Media Competition*. Kaggle, 2023.


