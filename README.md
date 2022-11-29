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
2. Numerical Optimozation
    - Working with SIMPLEPENDULUMOSCILLATIONDATA.csv
    - Learnings
      - understand optimization, and differentiate between convex and non-convex optimization
      - understand unconstrained and constrained optimizations
      - understand gradient descent methods
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
    - Working with Iris Dataset and Real Estate.csv
    - Learnings
      - have an overview of the basics of Machine Learning
      - understand the implementation of Train/Test Split
      - develop an understanding of Least Squares, Learning Curves
      - perform Linear Regression and KNN Regressor
      - have an understanding of Regularization of Linear Models
7. Linear Classification and LDA
    - Worked with MNIST and Fashion MNIST dataset
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
    - Working with UCI PIMA Indian Diabetes Dataset and Social Network Advertising dataset
    - Learnings
       - learn about Classification tasks in Machine learning
       - learn the appropriate performance metrics according to use case
10. Naive Bayes and Decision Trees
    - Learnings
       - understanding the basics of decision trees and Naive Bayes Algorithm
       - use multiple metrics which are popular with the decision tree algorithm and Naive Bayes 
11. Perceptron and Intro to SVM
    - Worked with Iris.csv
    - Learnings
       - understanding and experimenting with perceptrons and SVMs
12. Hard and Soft Margin and Kernel Support Vector Machines - SVM
    - Learnings
       - understand the basics of soft margin vs hard margin SVM
       - code a soft margin SVM and a hard margin SVM from scratch.
       - solve a problem using kernel ridge regression.
13. Ensemble Learning and Random Forests
    - Learnings
       - understand Ensemble learning and Ensemble methods
       - perform Voting Classifier and Bagging Classifier using Scikit-Learn package
       - understand the concept of Random Forest
       - perform classification using RandomForestClassifier
       - implement Gradient Boosting
14. Clustering
    - a dataset containing car sales
    - Learnings
      - understand the concept of clustering.
      - use different types of clustering methods like
          - K-Means,
          - Hieranchical and,
          - Gaussian Mixture Models.
      - compare the three different techniques on a standard datas
15. PCA
    - Learnings
       - Principal Component Analysis, Dimensionality reduction
16. Assocation Rule Mining ARM
    - Working wihtMArketing campaign dataset to perfomr CONSUMER BUYING BEHAVIOUR ANALYSIS
    - Learnings
       - perform exhasutive EDA
       - visualize the data distributions
       - understand and implement Assocation Rule Mining
17. Sentiment Analysis using linear classifiers and unsupervised clustering.
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
18. Credit Card Default Risk Analysis
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
19. Community detection PPI
    - Here the protein protein interaction in fly(Drosophila melanogaster) has been carried out and the dataset DM-LC.txt has been used
    - Learnings 
      - Community Detection in protein network using netwoekx for graph analysis
20. Essential Genes - Prediction of Essential Genes from Networks
    - to predict Essential Genes using the Protein network as the features of the STRING dataset
    - Learnings 
      - Get an understanding of the dataset
      - Build and analyze Networks (or Graphs)
      - Predict Essential Genes using the classification algorithm
21. Traffic Analysis and Prediction
    - worked with INRIX traffic dataset
    - Learnings 
      - convert the dataset to time series dataset
      - predict the flow of traffic at any junction
22. HR Analytics - Attrition
    - worked with IBM HR Analytics Employee Attrition & Performance Dataset
    - Learnings 
      - Get an understanding of the dataset.
      - Perform Extensive EDA and Visualizations
      - Handraft the raw data suitable for a ML problem
      - Predict(Classify) the employee Attrition based on employee performance
     
        |Index| Model  | classification Score |
        |------------------| --------------- | ------------- |
        |1.| Logistic Regression  |  0.53 |
        |2.| Naive Bayes  | 0.79 |
        |3.| SVM  | 0.87 | 
23. Software Bugs Detection
    - worked with Roundcube mail application Dataset
    - Learnings 
      - Get an understanding of the dataset.
      - Perform Extensive EDA and Visualizations
      - utilizing the fundamental building blocks of the NLP to classify the issues under appropriate categories based on the text body of the issue/ticket being raised.
     
        |Index| Model  | Target - "Defect Type Family using IEEE" | Target - "Defect Type Family using ODC"|
        |------------------| --------------- | ------------- | ------------- |
        |1.| Logistic Regression  |  0.23 | 0.33 |
        |2.| SVC  | 0.59 | 0.14 | 
24. Resume Classification
    - worked with Resume Recommendation Dataset
    - Learnings 
      - perform data preprocessing, EDA, feature extraction and NLP on the Resume dataset
      - perform multinomial Naive Bayes classification on the Resume dataset
25. Image classification using MLP and CNN
    - worked with German Traffic Sign Detection Benchmark (GTSDB) Dataset
    - Learnings 
      - load and extract features of images
      - implement simple neural network from Keras
      - implement CNN using Keras
26. Stock Prices Anomaly Detection
    - Using the S&P 500 stock prices data of different companies, we will perform a PCA based analysis.
    - Using the S&P 500 stock price index time series data, we will perform anomaly detection in the stock prices across the years.
    - Learnings 
      - perform PCA based stock analytics
      - analyze and create time series data
      - implement LSTM auto-encoders
      - detect the anomalies based on the loss
27. Banknote authentication - https://www.kaggle.com/code/jaina865/banknote-authentication-using-keras
28. Cancer Detection in CT Scan Images using CNN
    - Worked with CT images from cancer imaging archive (TCIA) with contrast and patient age.
    - Learnings 
      - The dataset is designed to allow for different methods to be tested for examining the trends in CT image data associated with using contrast and patient age.
      - The basic idea is to identify image textures, statistical patterns and features correlating strongly with these traits and possibly build simple tools for automatically classifying these images when they have been misclassified Data
