# Q1 Project ML
Gus Simanson &amp; Lalitendra Boyapati ML Q1 Project for Dr. Yilmaz's ML 1 Class

This project aims to predict NBA All-Star selections using machine learning models trained on player statistics from the 2010-2023 seasons. Data was scraped from stats.nba.com and basketball-reference.com, including 68 attributes such as points, assists, rebounds, and minutes played. After pre-processing, attribute selection techniques (CorrelationAttributeEval, CfsSubsetEval, InfoGainAttributeEval, and OneRAttributeEval) were applied to identify the most relevant features. The dataset was split into training, testing, and validation sets, and synthetic minority oversampling (SMOTE) was used to address class imbalance. Four classification models—Naive Bayes, J48, OneR, and Logistic Regression—were tested. 

# How to Run
Included within this repository is the Python webscraping code used to gather our player data. The preprocessing, attribute selection, and classification was done of WEKA.
