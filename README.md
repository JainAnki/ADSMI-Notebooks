# ADSMI-Notebooks
# ADSMI-Notebooks
1. General preprocessing techniques
    1. Exploratory data analysis
        - checking for the number of rows and columns
        - summary of the dataset
        - statistical description of the features
        - check for the duplicate values
        - Show the top 5 and the last 5 rows of the data
        - check for the null values, and handle them if any
    2. Data Preparation
        - Create new features
        - Create possible new features based on the existing data
        - Identify the features, target from the given set of attributes and split the data into train, test
        - Normalization *StandardScaler or MinMaxScaler(
    3. Visualization
        - Plot the distribution of all the variables as histograms
        - Correlations between variables as heatmap
        - Analyze the results between target and other features (Hint: Pair plot)
    6. 
2. Numerical Optimozation
    - Learnings
      - understand optimization, and differentiate between convex and non-convex optimization
      - understand unconstrained and constrained optimizations
      - understand gradient descent methods
      - Working with SIMPLEPENDULUMOSCILLATIONDATA.csv
3. Probability and Statistics for Data Science
   - Learnings
      - know a general idea on how probability plays an important role in data science.
      - understand sampling distributions, sampling mean and standard-deviaton. Central limits theorem.
      - use hypothesis testing: T Test, Z Test, Chi-Square Test
      - understand how confidence intervals are used.
      - know more about correlation functions, use parameter estimation using MLE and Bayesian methods.
4. California Housing Price Data Set - Linear Regression with Gradient Descent
    - Loading and looking at data and implementing
        - Cost function
        - Gradient descent variants
5. KNN_Naive Bayes 
    - Learnings
      - understand k-Nearest Neighbors and implement from scratch and implement k-Nearest Neighbors and verifying using sklearn library
      - understand terms related to Bayesian Inference
6. Regression Models
    - Learnings
      - have an overview of the basics of Machine Learning
      - understand the implementation of Train/Test Split
      - develop an understanding of Least Squares, Learning Curves
      - perform Linear Regression and KNN Regressor
      - have an understanding of Regularization of Linear Models
      - Working with Iris Dataset and Real Estate.csv
7. Linear Classification and LDA
    - Learnings
      - understand and code a logistic regression algorithm.
      - understand the basics of Linear Discriminant Analysis (LDA)
      - use toy datasets for binary classification
      - use the standard MNIST dataset for multiclass classification.
      - LDA trained with different number of features:

      |Number of features| Dataset  | accuracy |
      |------------------| --------------- | ------------- |
      |1| MNIST | 40.53% |
      || Fashion MNIST | 47.45% |
      |2| MNIST | 54.80% |
      || Fashion MNIST |  59.34% |
      |5| MNIST |  82.99% |
      || Fashion MNIST |  74.02% |
      |7| MNIST | 86.9% |
      || Fashion MNIST | 78.46% |
      |9| MNIST | 88.67% |
      || Fashion MNIST | 82.41% |
      |10| MNIST | 92.55% |
      || Fashion MNIST | 84.12% |
      
8. Model Selection and Cross Validation
    - Learnings
       - Understand the different Cross-validation of data
       - Understand the importance and implementation of Cross-validation of ML models
       - Develop an understanding of Model-Selection
       - Akaike's Information Criteria (AIC) and Bayesian Information Criteria (BIC)

       
9. Evaluation of Performance Metrics
    - Learnings
       - learn about Classification tasks in Machine learning
       - learn the appropriate performance metrics according to use case
       - Working with UCI PIMA Indian Diabetes Dataset and Social Network Advertising dataset
10. Naive Bayes and Decision Trees
    - Learnings
       - understanding the basics of decision trees and Naive Bayes Algorithm
       - use multiple metrics which are popular with the decision tree algorithm and Naive Bayes 
11. Perceptron and Intro to SVM
    - Learnings
       - understanding and experimenting with perceptrons and SVMs
       - Worked with Iris.csv
12. Sentiment Analysis using linear classifiers and unsupervised clustering.
    - a dataset containing amazon review information along with ratings
    - Learnings
      - undertake several important steps like cleaning the data and normalizing the data points.
      - do sentiment classification.
      - compare between different types of classification methods and their pros and cons.
      - compare between supervised and unsupervised (clustering) techniques
      
      |Index| Model  | accuracy | f1_score  |
      |------------------| --------------- | ------------- | ------------- |
      |1.| K-Nearest Neighbour Classifier  | 0.8369032824036845 | 0.9101019462465245  |
      |2.| Support Vector Machines (SVM) Classifier  |   |
      |2a.| Hard Margin  | 0.8385846918634403 |0.91220675944334 |
      |2b.| Soft Margin  | 0.8385846918634403 |0.91220675944334 |
      |2c.| Kernel SVM   | 0.8385115871043205 |0.9121635055071773 |
      |3.| Decision Tree Classifier | 0.7336062577673806 |  0.8409150440932507 |
      |4.|Ensemble Classifier 
      |4a.|KNeighborsClassifier, GaussianNB,LogisticRegression|0.8371957014401638 |0.911363184079602|

10. Credit Card Default Risk Analysis
    - A dataset containing credit defaulters data
    - Learnings 
      - Understanding different classification techniques like:
        - Logistic Regression
        - Support Vector Classifier
        - Multi Layer Perceptron
        - Random Forests        

        |Index| Model  | accuracy | f1_score  |
        |------------------| --------------- | ------------- | ------------- |
        |1.| Logistic Regression  |  0.82 | 0.89  |
        |2.| SVM  | 0.82 |0.89 |
        |3.| Multi Layer Perceptron  | 0.80 |0.88 | 
        |4.| Random Forests | 0.82 |0.89 |


