# import nltk
import pandas as pd
import re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# nltk.download('stopwords')

# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = ENGLISH_STOP_WORDS


# -----------------------------
# Preprocessing function
# -----------------------------
# def preprocess_text(content):
#     if not isinstance(content, str):
#         return ""
#     content = re.sub('[^a-zA-Z]', ' ', content)
#     content = content.lower().split()
#     content = [ps.stem(word) for word in content if word not in stop_words]
#     return ' '.join(content)

def preprocess_text(content):
    if not isinstance(content, str):
        return ""
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [word for word in content if word not in ENGLISH_STOP_WORDS]
    return ' '.join(content)



def load_and_prepare_data():
    df = pd.read_csv("news.csv", usecols=['Statement', 'Label'], low_memory=False, encoding='latin1')
    df = df.dropna(subset=['Statement', 'Label'])
    df['processed'] = df['Statement'].apply(preprocess_text)
    return df


def vectorize_and_split(df):
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['processed'])
    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, vectorizer


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# Main logic
# -----------------------------
df = load_and_prepare_data()
X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df)
model = train_model(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n📊 Model Evaluation on Test Set")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)
labels = ["Real", "Fake"]

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------
# Bar Chart (Precision, Recall, F1-score)
# -----------------------------
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0,1])

metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
}

labels = ["Real News", "Fake News"]
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(7,5))
rects1 = ax.bar(x - width, metrics["Precision"], width, label="Precision")
rects2 = ax.bar(x, metrics["Recall"], width, label="Recall")
rects3 = ax.bar(x + width, metrics["F1-score"], width, label="F1-score")

ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_title("Performance Metrics")

# Show values on top of bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.savefig("metrics_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()


# Save model and vectorizer
# import joblib
# joblib.dump(model, "fake_news_model.pkl")
# joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Step 5: Predict from terminal input
# while True:
#     print("\n🔎 Enter a news statement to classify (or type 'exit' to quit):")
#     input_text = input(">> ")
#     if input_text.lower() == 'exit':
#         break

#     processed = preprocess_text(input_text)
#     vector_input = vectorizer.transform([processed])
#     prediction = model.predict(vector_input)[0]

#     result = "🟥 FAKE NEWS" if prediction == 1 else "🟩 REAL NEWS"
#     print(f"Prediction: {result}")



