## Problem Statement

This project focuses on building a deep learning model to predict high-frequency price changes of two unidentified US stocks. The model has an accuracy of 73% in the unseen testing dataset.

The objective is to train a binary classifier using a dataset containing historical limit order book data to predict the direction of midprice changes (up or down) in the financial market.

The data is split into two sets:

- **Data_A.csv**: Contains labeled data with features related to bid and ask prices, volumes, and previous price changes.
- **Data_B_nolabels.csv**: Contains similar features but without the labels, which the model must predict.

## Deep Learning Methodologies

The solution utilizes various deep learning methodologies to develop a robust binary classifier:

1. **Feedforward Neural Network (FNN):** A basic architecture using fully connected layers to learn patterns in the input features and predict the midprice change direction.
2. **Recurrent Neural Network (RNN):** Used to capture sequential dependencies and temporal patterns in the data, leveraging the historical features provided.
3. **Convolutional Neural Network (CNN):** Applies convolutional layers to detect local patterns in the input feature matrix, which may represent meaningful price movements and volume changes.
4. **Regularization Techniques:** Techniques like dropout and L2 regularization are implemented to prevent overfitting.
5. **Data Normalization and Feature Scaling:** Ensuring consistent input data by normalizing and scaling the features, vital for improving model convergence and accuracy.

The model is trained using a portion of the labeled dataset, while a separate validation set is used to fine-tune hyperparameters and prevent overfitting. The final model is used to generate predictions for the unlabeled dataset.
