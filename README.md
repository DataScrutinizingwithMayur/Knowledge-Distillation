# Knowledge Distillation Project

## Overview
This project demonstrates the application of Knowledge Distillation, a model compression technique, to efficiently transfer knowledge from a larger Convolutional Neural Network (CNN) model (teacher) to a smaller Feedforward Neural Network (student). By doing so, we aim to maintain high model performance while reducing computational costs and model size, making the model more suitable for deployment in resource-constrained environments.

## Motivation
Knowledge Distillation (KD) is a critical step towards efficient machine learning, allowing smaller models to learn from the complex representations captured by larger models without the need for extensive computational resources. This project is inspired by techniques discussed in [Model Compression: A Critical Step Towards Efficient Machine Learning](https://www.dailydoseofds.com/model-compression-a-critical-step-towards-efficient-machine-learning/), focusing on leveraging KD for practical model compression.

## Methodology
- **Teacher Model**: A Convolutional Neural Network (CNN), trained on a given dataset to achieve high accuracy. This model serves as the source of knowledge.
- **Student Model**: A smaller, less complex Feedforward Neural Network designed to learn the distilled knowledge from the teacher model.
- **Distillation Process**: Utilizes the Kullback-Leibler (KL) Divergence loss to measure the difference between the output distributions (logits) of the teacher and student models, guiding the student model to mimic the teacher's behavior.

## Results
The distillation process effectively compresses the knowledge from the teacher model into the student model with minimal loss in accuracy.

- **Teacher Model Performance**:
  - Epoch 1: Loss: 0.2164, Accuracy: 96.50%
  - Epoch 2: Loss: 0.0745, Accuracy: 97.58%
  - Epoch 3: Loss: 0.0572, Accuracy: 98.37%
  - Epoch 4: Loss: 0.0473, Accuracy: 97.73%
  - Epoch 5: Loss: 0.0391, Accuracy: 98.59%

- **Student Model Performance**:
  - Epoch 1: Loss: 0.0675, Accuracy: 97.12%
  - Epoch 2: Loss: 0.0600, Accuracy: 97.34%
  - Epoch 3: Loss: 0.0539, Accuracy: 96.38%
  - Epoch 4: Loss: 0.0503, Accuracy: 97.24%
  - Epoch 5: Loss: 0.0469, Accuracy: 97.31%

## Conclusion
The Knowledge Distillation technique, specifically using KL Divergence for loss calculation, proves to be an effective strategy for model compression. The student model, despite its simplicity compared to the CNN teacher model, achieves comparable accuracy, showcasing the potential of KD in creating efficient and deployable machine learning models.

## How to Use
1. Clone the repository.
2. Open the Jupyter Notebook containing the knowledge distillation code.
3. Follow the notebook instructions to train the teacher and student models on your dataset.

## Requirements
- Python 3.x
- PyTorch
- Other dependencies listed in `requirements.txt`.

For more detailed information on the methodology and results, please refer to the accompanying Jupyter Notebook.
