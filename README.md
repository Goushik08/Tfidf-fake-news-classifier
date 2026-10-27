# Tfidf-fake-news-classifier
Fake news detection using TF-IDF and 8 scikit-learn models (Linear SVM, SGD, Passive Aggressive, Logistic Regression and more). Best model: Linear SVM at 93.6% accuracy.
# Fake News Detection with TF-IDF

Classifies news articles as **FAKE** or **REAL** using a TF-IDF text representation and compares eight scikit-learn models.

**Best model: Linear SVM, 93.6% accuracy** on a held-out test set of 2,091 articles.

## Results

| Rank | Model | Accuracy | FAKE Recall | REAL Recall | Macro F1 |
|---|---|---|---|---|---|
| 1 | Linear SVM | 93.6% | 0.96 | 0.91 | 0.94 |
| 2 | SGD Classifier | 93.6% | 0.95 | 0.92 | 0.94 |
| 3 | Passive Aggressive | 93.3% | 0.94 | 0.92 | 0.93 |
| 4 | Logistic Regression | 91.3% | 0.95 | 0.88 | 0.91 |
| 5 | Random Forest | 90.8% | 0.92 | 0.89 | 0.91 |
| 6 | Complement NB | 85.2% | 0.72 | 0.97 | 0.85 |
| 7 | KNN | 83.6% | 0.73 | 0.94 | 0.83 |
| 8 | Decision Tree | 81.1% | 0.80 | 0.82 | 0.81 |

![Model comparison output](images/model_comparison.png)

Linear SVM confusion matrix (rows = actual, columns = predicted):

| | Predicted FAKE | Predicted REAL |
|---|---|---|
| **Actually FAKE** | 969 | 39 |
| **Actually REAL** | 94 | 989 |

Linear models perform best on sparse, high-dimensional text. Complement NB and KNN miss about 27% of fake articles despite decent overall accuracy.

## Project Structure

```
tfidf-fake-news-classifier/
├── fake_news_classifier.py    # Compares 8 models (main script)
├── baseline_naive_bayes.py    # Simple MultinomialNB baseline
├── requirements.txt
├── images/
│   └── model_comparison.png
└── news.csv                   # You provide this (not included)
```

## Dataset

The scripts expect a file named `news.csv` in the project root with these columns:

| Column | Description |
|---|---|
| `Unnamed: 0` | Index column (dropped by the script) |
| `title` | Article headline |
| `text` | Article body (used as the input) |
| `label` | `FAKE` or `REAL` (the target) |

The results above were produced on a dataset of 6,335 articles, split 67% train / 33% test with `random_state=53`.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/<your-username>/tfidf-fake-news-classifier.git
cd tfidf-fake-news-classifier

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

Place `news.csv` in the project folder, then run:

```bash
# Compare all eight models
python fake_news_classifier.py

# Or run only the Naive Bayes baseline
python baseline_naive_bayes.py
```

The comparison script prints accuracy, a confusion matrix and a classification report for every model, then a ranked summary and a prediction on a sample article using the best model.

The code also runs cell by cell in Jupyter Notebook (each `# Step N` comment is one cell).

## How It Works

1. Load `news.csv` and drop the unused index column.
2. Split into train and test sets (33% test).
3. Convert text to numbers with `TfidfVectorizer(stop_words='english')`. The vocabulary is learned from the training set only, then applied to the test set.
4. Train each model on the TF-IDF training matrix.
5. Evaluate on the test set and rank by accuracy.

## Limitations

- Results come from a single train/test split.
- Models use mostly default hyperparameters (no tuning).
- The classifier learns writing patterns from this dataset; it does not verify facts and may not generalize to other sources without retraining.

## Possible Improvements

- 5-fold cross-validation for a more reliable ranking
- Hyperparameter tuning (for example `C` for Linear SVM) with `GridSearchCV`
- Bigrams and a `max_df` cap in `TfidfVectorizer`
- Testing on articles from outside this dataset

## Related

A hashing-vectorizer version of this project is in `hashing-fake-news-classifier`. TF-IDF scored about 0.6 points higher on average (best model 93.6% vs 93.1%), while hashing uses fixed memory and needs no stored vocabulary.
