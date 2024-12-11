# Sentiment Analysis on Product Reviews
## Overview
This project implements a Sentiment Analysis pipeline using various machine learning models to classify product reviews into Positive, Neutral, and Negative categories. The dataset consists of user reviews, and the analysis leverages natural language processing (NLP) and machine learning techniques.

## Features
- Data preprocessing: Cleaning and transforming raw review data.
- Sentiment categorization based on review scores.
- Text vectorization using TF-IDF (Term Frequency-Inverse Document Frequency).
- Model training and evaluation with the following classifiers:
   - Linear SVM
   - Random Forest
   - Multinomial Naive Bayes
   - Bernoulli Naive Bayes
   - Logistic Regression
- Performance metrics: Accuracy, precision, recall, and F1-score.

## Dataset
The dataset, Reviews.csv, contains product reviews with scores ranging from 1 to 5. The project uses a subset of 60,000 reviews for training and evaluation.

## Sentiment Categories
Positive: Scores of 4 or 5.
Neutral: Score of 3.
Negative: Scores of 1 or 2.

## Requirements
Install the required Python libraries using the command:
pip install -r requirements.txt

## Dependencies
- pandas
- scikit-learn
- nltk

## Pipeline
1. Data Preprocessing:
- Handle missing values and convert scores to integer values.
- Categorize scores into sentiment labels (Positive, Neutral, Negative).
- Clean the text by removing non-alphabetic characters, stopwords, and converting to lowercase.
2. Feature Extraction:
- Use TF-IDF Vectorization to convert text data into numerical features.
- Train-Test Split:
- Split the data into 80% training and 20% testing.
3. Model Training:
- Train and evaluate multiple classifiers.
4. Evaluation:
- Evaluate each model's performance using accuracy and a classification report.

## Models Used
- Linear SVM: A powerful linear classifier.
- Random Forest: An ensemble of decision trees.
- Multinomial Naive Bayes: Suitable for discrete features.
- Bernoulli Naive Bayes: Assumes binary feature presence/absence.
- Logistic Regression: Effective for linear decision boundaries.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/UsmanAli90/Amazon-Sentiment-Analysis
```
2. Navigate to the project directory:
```bash
cd Amazon-Sentiment-Analysis
```
3. Place the dataset (Reviews.csv) in the project directory.
4. Run the script:
```bash
python sentiment_analysis.py
```
## Results
- Models are evaluated on accuracy and detailed classification metrics.
- Comparison of different classifiers is provided.

## Future Enhancements
- Add deep learning models such as LSTM or BERT for improved accuracy.
- Implement sentiment visualization tools (e.g., word clouds, bar plots).
- Explore other feature extraction techniques like Word2Vec or GloVe.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
