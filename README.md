# IMDB Sentiment Analysis Web App

A web application that performs sentiment analysis on movie reviews using machine learning. The app analyzes text input to determine whether a review is positive or negative, with special handling for negation (e.g., "not good" correctly predicts negative sentiment).

## Tech Stack

- **Python 3.13** - Core programming language
- **scikit-learn** - Machine learning library for TF-IDF vectorization and LogisticRegression
- **Flask** - Web framework for the application
- **TF-IDF** - Text feature extraction technique
- **NLTK** - Natural language processing for text preprocessing
- **pandas** - Data manipulation and analysis
- **matplotlib** - Chart generation for analytics
- **joblib** - Model serialization

## Features

- **Sentiment Prediction**: Analyze text input and predict positive or negative sentiment with confidence scores
- **Web Interface**: Clean, responsive web UI built with Flask and Jinja2 templates
- **Analytics Page**: View dataset statistics and interactive bar chart showing sentiment distribution
- **Negation Handling**: Advanced text preprocessing that correctly handles negations (e.g., "not bad" → negative)
- **Model Performance**: 89% accuracy on test data with balanced performance across classes



### What It Provides

- **Instant Sentiment Analysis**: Real-time prediction of text sentiment with probability scores
- **Educational Tool**: Learn about natural language processing and machine learning through a practical web application
- **Data Insights**: Understand sentiment patterns in text data through the analytics dashboard
- **Negation-Aware Processing**: Accurately handles complex language constructs like negations
- **High Accuracy**: 89% accuracy model trained on large IMDB dataset for reliable predictions
- **Web-Based Accessibility**: No installation required for end-users, accessible via any web browser



### ML Pipeline Overview

1. **Data Loading**: The model was trained on the IMDB movie reviews dataset (40,000 reviews)

2. **Text Preprocessing**:
   - Convert to lowercase
   - Handle contractions (don't → do not)
   - Remove punctuation and normalize whitespace
   - Apply lemmatization using WordNet
   - **Negation Handling**: Preserve negation words and mark following words (e.g., "not good" becomes "not NOT_good")

3. **Feature Extraction**:
   - TF-IDF vectorization with unigrams and bigrams
   - Limited to top 10,000 features for efficiency

4. **Model Training**:
   - LogisticRegression with hyperparameter tuning via GridSearchCV
   - Optimized parameters: C=1.0, solver='lbfgs'
   - 5-fold cross-validation for robust evaluation

5. **Prediction**:
   - Input text is preprocessed using the same pipeline
   - Transformed to TF-IDF features
   - Model predicts sentiment (0=negative, 1=positive)
   - Returns prediction with confidence probability

### Model Performance
- **Accuracy**: 89.05%
- **F1-Score**: 89.05%
- **Cross-validation F1**: 89.18%
- Balanced performance across positive and negative classes




