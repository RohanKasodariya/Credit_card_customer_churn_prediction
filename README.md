# Credit Card Customer Churn Prediction

This repository contains a project aimed at predicting the likelihood of customer churn in a credit card company using various machine learning techniques. Customer churn is a significant challenge for businesses, and predicting it can help in taking proactive measures to retain customers.

## 🗂 Project Overview

The goal of this project is to build a predictive model that estimates the probability of a customer leaving the credit card company based on various behavioral and demographic features. This project demonstrates the use of data preprocessing, feature engineering, and multiple classification models to address a binary classification problem.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Modeling Approach](#-modeling-approach)
- [Prerequisites](#-prerequisites)
- [Usage](#-usage)
- [Future Work](#-future-work)
- [Contact](#-ontact)

## 📊 Dataset

The dataset used in this project contains the following features:
- **CustomerID**: Unique ID for each customer.
- **Credit Score**: Credit score of the customer.
- **Geography**: The country where the customer resides.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been with the company.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of products the customer has with the company.
- **HasCrCard**: Whether the customer has a credit card (0 or 1).
- **IsActiveMember**: Whether the customer is an active member (0 or 1).
- **EstimatedSalary**: Estimated salary of the customer.
- **Exited**: Target variable indicating whether the customer has churned (0 or 1).

## 🧠 Modeling Approach

The project employs various machine learning techniques to predict customer churn:
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
- **Exploratory Data Analysis (EDA)**: Understanding data distributions, correlations, and identifying key factors influencing churn.
- **Model Building**: Implementing multiple classification models including Logistic Regression, Random Forest, and Gradient Boosting.
- **Model Evaluation**: Evaluating models using accuracy, precision, recall, F1-score, and ROC-AUC.

### Model Details:
- **Baseline Model**: Logistic Regression as a simple baseline.
- **Advanced Models**: Random Forest, Gradient Boosting, and Neural Networks.
- **Hyperparameter Tuning**: Utilizing Grid Search and Random Search to find the best hyperparameters for each model.

## 🛠 Prerequisites

To run this notebook, you need to have the following libraries installed:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost (if used)
- TensorFlow/Keras (if used)


## 🖥 Usage

1. **Data Preparation**: Follow the steps in the notebook for data loading and preprocessing.
2. **Model Training**: Train different models by executing the respective cells in the notebook.
3. **Evaluation**: Evaluate the models on the test set and visualize the performance metrics.

## 🔍 Future Work

Some potential improvements for this project include:
- Implementing more sophisticated feature engineering techniques.
- Exploring deep learning models like LSTM or CNN for time-series or sequential data if applicable.
- Developing a customer segmentation strategy to identify high-risk churn groups.

## 📫 Contact

- **Email**: [kasodariya.r@northeastern.edu](mailto:kasodariya.r@northeastern.edu)
- **LinkedIn**: [Rohan Kasodariya](https://www.linkedin.com/in/rohankasodariya/)
- **GitHub**: [RohanKasodariya](https://github.com/RohanKasodariya)

Feel free to reach out if you have any questions or suggestions!

---

Thanks for checking out this project! 😊
