# Food Classification Deep-Learning Model

## Introduction

This project is a deep learning model for image classification of a set of 10 food items. It uses the ResNet-50 pre-trained on ImageNet database.

## Dataset

The dataset this model was trained on is composed of between 3000 images of a 10 standalone food items scraped from various sources (Pinterest, tumblr, reddit, etc). Each category is composed of 300 images to keep the dataset balanced. Categories of food included in this dataset are:

- Cheeseburger
- Cake
- Cookie
- Fries
- Hotdog
- Pizza
- Salad
- Shrimp
- Steak
- Sushi

## Requirements
This project requires the following packages:

torch>=1.10.0
torchvision>=0.11.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
numpy>=1.19.0
Pillow>=8.0.0
tqdm>=4.50.0

## Results
After training the model for 10 epochs with a batch size of 32 and a learning rate of 0.001, we achieved an accuracy of 93.33% on the train set, 93.67% on the validation set, and 92.83% on the test set(as of 13/11/2025). Class-wise accuracies following test are as follows:

| Class  | Accuracy |
 --- | --- |
| Burger |  98.18% |
| Cake |  81.48% |
| Cookie |  95.31% |
| Fries |  91.53% |
| Hotdog |  95.31% |
| Pizza |  89.36% |
| Salad |  96.83% |
| Shrimp |  90.91% |
| Steak |  92.96% |
| Sushi |  94.74% |



## License
This project is licensed under the [MIT License](LICENSE).

 
