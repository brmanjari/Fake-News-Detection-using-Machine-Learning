# Fake-News-Detection-using-Machine-Learning

A machine learning project that detects fake news articles using natural language processing and various classification algorithms. The system analyzes textual features to distinguish between legitimate and fabricated news content.
Overview
In the era of information overload, distinguishing between authentic and fake news has become increasingly challenging. This project implements multiple machine learning algorithms to automatically classify news articles as real or fake based on their textual content.
The system uses natural language processing techniques to extract meaningful features from news articles and employs various classification models to make predictions with high accuracy.
Features

Multiple ML Algorithms: Implements Random Forest, SVM, Logistic Regression, and Naive Bayes classifiers
Text Preprocessing: Comprehensive text cleaning, tokenization, and feature extraction
Feature Engineering: TF-IDF vectorization and n-gram analysis
Model Comparison: Side-by-side performance evaluation of different algorithms
Interactive Prediction: Real-time fake news detection for new articles
Visualization: Performance metrics and data distribution plots
Cross-validation: Robust model evaluation using k-fold cross-validation

Dataset
The project uses a comprehensive fake news dataset containing:

Size: 20,800+ news articles
Features: Title, text content, author, publication date
Labels: Binary classification (Real/Fake)
Sources: Mix of reliable news outlets and known fake news websites

Data Distribution

Real News: 10,413 articles (50.1%)
Fake News: 10,387 articles (49.9%)

Installation
Prerequisites

Python 3.8 or higher
pip package manager

Project Structure
fake-news-prediction/
│
├── data/
│   ├── raw/                    # Raw dataset files
│   ├── processed/              # Cleaned and preprocessed data
│   └── sample/                 # Sample data for testing
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── fake_news_detector.py
│
├── models/
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── vectorizer.pkl
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_prediction.py
│
├── requirements.txt
├── train_model.py
├── predict.py
├── README.md
└── LICENSE
Technologies Used

Python: Primary programming language
scikit-learn: Machine learning algorithms and evaluation
NLTK: Natural language processing and text preprocessing
Pandas: Data manipulation and analysis
NumPy: Numerical computing
Matplotlib/Seaborn: Data visualization
Jupyter Notebook: Interactive development environment

Methodology

Data Collection: Gathered labeled dataset of real and fake news articles
Data Preprocessing:

Text cleaning (removing special characters, URLs, extra spaces)
Tokenization and lowercasing
Stop word removal
Stemming/Lemmatization


Feature Engineering:

TF-IDF vectorization
N-gram analysis (unigrams, bigrams, trigrams)
Text length and readability features


Model Training:

Split data into training/validation/test sets
Train multiple classification algorithms
Hyperparameter tuning using GridSearchCV


Evaluation:

Cross-validation
Confusion matrix analysis
ROC curve and AUC metrics



Future Improvements

 Implement deep learning models (LSTM, BERT)
 Add real-time web scraping for live news analysis
 Create a web application interface
 Incorporate additional features (source credibility, author reputation)
 Multilingual support
 Integration with news APIs
