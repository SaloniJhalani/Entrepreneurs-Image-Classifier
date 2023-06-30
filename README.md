# Entrepreneurs Image Classifier

This project focuses on creating an image classification model to classify images of five entrepreneurs collected from the internet. The goal is to develop a model that can accurately identify and classify the images of these entrepreneurs. The following steps were undertaken in this project:

1. Data Collection:
* Images of five entrepreneurs were collected from the internet.
* The collected images were stored in an appropriate directory structure for further processing.

2. Data Preprocessing:
* The collected images were cleaned and filtered to remove any irrelevant or noisy images.
* Image transformations, such as resizing, cropping, and normalization, were applied to ensure consistency and optimal model performance.
* The data was split into training and testing sets for model evaluation.

3. Model Training and Evaluation:
* Various classification models, including Logistic Regression, Random Forest, and Support Vector Machines (SVM), were implemented.
* The models were trained on the preprocessed image data.
* Model performance metrics, such as accuracy, precision, recall, and F1-score, were calculated to evaluate the effectiveness of each model.

4. Model Selection and Deployment:
* Based on the evaluation results, the best-performing model was selected for deployment.
* The chosen model was saved and prepared for use in production environments.

## Technologies Used:

* Python: Programming language used for data preprocessing, model training, and evaluation.
* OpenCV: Library utilized for image processing and transformations.
* Scikit-learn: Framework employed for implementing and evaluating classification models.
* Jupyter Notebook: Interactive development environment utilized for code execution and documentation.




## Table of Contents
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Website Link](#website-link)
- [Implementation Details](#implementation-details)
    - [Methods Used](#methods-used)
    - [Technologies](#technologies)
    - [Python Packages Used](#python-packages-used)
- [Steps Followed](#steps-followed)
- [Results and Evaluation Criterion](#results-and-evaluation-criterion)
- [Future Improvements](#future-improvements)

  
## Project Overview
This project focuses on developing a food delivery time prediction model. The goal is to estimate the time it takes for food to be delivered to customers accurately. By accurately predicting delivery times, food delivery platforms can enhance customer experience, optimize delivery logistics, and improve overall operational efficiency.

## Data Source
The dataset used for this project can be obtained from [here](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset).

It contains relevant information such as order details, location, city, delivery person information, weather conditions and actual delivery times.

## Website Link

A web-based demonstration of the food delivery time prediction model can be accessed from this [link](https://food-delivery-time-prediction.streamlit.app).

## Implementation Details

### Methods Used
* Machine Learning
* Data Cleaning
* Feature Engineering
* Regression Algorithms

### Technologies
* Python
* Jupyter
* streamlit

### Python Packages Used
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* xgboost

## Steps Followed

1. Data Collection: Gathered the food delivery dataset from the provided data source.
2. Data Preprocessing: Performed data cleaning to handle missing values, outliers, and inconsistencies in the dataset. Conducted feature engineering to extract relevant features for the prediction model.
3. Model Development: Utilized regression algorithms to train a food delivery time prediction model. Explored different models such as linear regression, decision trees, random forests, xgboost to identify the best-performing model.
4. Model Evaluation: Evaluated the performance of the models using appropriate metrics such as mean squared error (MSE),root mean squared error (RMSE) and R2 score.
5. Deployment: Deployed the food delivery time prediction model as a standalone application for real-time predictions.

## Results and Evaluation Criterion

Based on the evaluation results, the best-performing model was **XGBoost** with R2 score of **0.82**

## Future Improvements

Here are some potential areas for future improvements in the project:

* Incorporate more features related to delivery partners, weather conditions, or traffic patterns to enhance prediction accuracy.
* Conduct more comprehensive data analysis to identify additional patterns or correlations that can contribute to better predictions.
* Fine-tune the model parameters to potentially improve performance.






