import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.pipeline import Pipeline


def train_system():

    # =====================================================
    # 1. CREATE FOLDERS
    # =====================================================

    os.makedirs("models", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    # =====================================================
    # 2. FILE PATHS
    # =====================================================

    true_path = "data/True.csv"
    fake_path = "data/Fake.csv"

    print("Loading dataset...")

    # =====================================================
    # 3. CHECK FILES
    # =====================================================

    if not os.path.exists(true_path):
        print("ERROR: data/True.csv not found!")
        return

    if not os.path.exists(fake_path):
        print("ERROR: data/Fake.csv not found!")
        return

    # =====================================================
    # 4. LOAD DATASET
    # =====================================================

    try:
        true_df = pd.read_csv(
        true_path,
        sep="\t",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

        fake_df = pd.read_csv(
        fake_path,
        sep="\t",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

    except UnicodeDecodeError:

        print("UTF-8 failed. Trying latin1...")

        true_df = pd.read_csv(
            true_path,
            encoding="latin1"
        )

        fake_df = pd.read_csv(
            fake_path,
            encoding="latin1"
        )

    # =====================================================
    # 5. CLEAN COLUMN NAMES
    # =====================================================

    true_df.columns = (
        true_df.columns
        .str.strip()
        .str.lower()
    )

    fake_df.columns = (
        fake_df.columns
        .str.strip()
        .str.lower()
    )

    print("\nTrue.csv columns:")
    print(true_df.columns.tolist())

    print("\nFake.csv columns:")
    print(fake_df.columns.tolist())

    # =====================================================
    # 6. CHECK REQUIRED COLUMNS
    # =====================================================

    if "text" not in true_df.columns and "title" not in true_df.columns:
        print("\nERROR: True.csv does not contain title/text columns.")
        return

    if "text" not in fake_df.columns and "title" not in fake_df.columns:
        print("\nERROR: Fake.csv does not contain title/text columns.")
        return

    # =====================================================
    # 7. ADD LABELS
    # =====================================================

    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    # =====================================================
    # 8. COMBINE DATA
    # =====================================================

    df = pd.concat(
        [true_df, fake_df],
        ignore_index=True
    )

    print("\nDataset Size:", df.shape)

    # =====================================================
    # 9. CREATE CONTENT
    # =====================================================

    if "title" in df.columns and "text" in df.columns:

        df["content"] = (
            df["title"].fillna("").astype(str)
            + " "
            + df["text"].fillna("").astype(str)
        )

    elif "text" in df.columns:

        df["content"] = (
            df["text"]
            .fillna("")
            .astype(str)
        )

    else:

        df["content"] = (
            df["title"]
            .fillna("")
            .astype(str)
        )

    # =====================================================
    # 10. REMOVE EMPTY ARTICLES
    # =====================================================

    df = df[
        df["content"].str.strip() != ""
    ]

    print("\nFinal Dataset Size:", df.shape)

    # =====================================================
    # 11. CLASS DISTRIBUTION
    # =====================================================

    print("\nClass Distribution:")
    print(df["label"].value_counts())

    df["label"].value_counts().plot(kind="bar")

    plt.title("Real vs Fake News Distribution")
    plt.xlabel("News Type")
    plt.ylabel("Number of Articles")

    plt.tight_layout()

    plt.savefig(
        "notebooks/distribution.png"
    )

    plt.close()

    # =====================================================
    # 12. FEATURES AND LABEL
    # =====================================================

    X = df["content"]
    y = df["label"]

    # =====================================================
    # 13. TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "\nTraining samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    # =====================================================
    # 14. MODELS
    # =====================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "Naive Bayes":
            MultinomialNB(),

        "Linear SVM":
            CalibratedClassifierCV(
                LinearSVC(),
                cv=3
            )
    }

    # =====================================================
    # 15. BEST MODEL VARIABLES
    # =====================================================

    best_model = None
    best_score = 0
    best_name = ""

    best_metrics = {}

    # =====================================================
    # 16. TRAIN MODELS
    # =====================================================

    for name, classifier in models.items():

        print("\n" + "=" * 60)
        print("Model:", name)
        print("=" * 60)

        pipeline = Pipeline([

            (
                "tfidf",

                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    stop_words="english"
                )
            ),

            (
                "classifier",
                classifier
            )
        ])

        print("Training...")

        pipeline.fit(
            X_train,
            y_train
        )

        # =================================================
        # PREDICTION
        # =================================================

        y_pred = pipeline.predict(
            X_test
        )

        # =================================================
        # METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label="FAKE"
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label="FAKE"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label="FAKE"
        )

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        # =================================================
        # SELECT BEST MODEL
        # =================================================

        if f1 > best_score:

            best_score = f1

            best_model = pipeline

            best_name = name

            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    # =====================================================
    # 17. FINAL RESULT
    # =====================================================

    print("\n" + "=" * 60)

    print(
        f"BEST MODEL: {best_name}"
    )

    print(
        f"Accuracy : {best_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: {best_metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : {best_metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score : {best_metrics['f1'] * 100:.2f}%"
    )

    print("=" * 60)

    # =====================================================
    # 18. SAVE MODEL
    # =====================================================

    model_path = "models/fake_news_model.pkl"

    joblib.dump(
        best_model,
        model_path
    )

    print("\nMODEL SAVED SUCCESSFULLY!")

    print(
        f"Model saved at: {model_path}"
    )

    # =====================================================
    # 19. TEST PREDICTION
    # =====================================================

    test_news = [
        "The government announced a new policy after a cabinet meeting."
    ]

    prediction = best_model.predict(
        test_news
    )[0]

    probability = best_model.predict_proba(
        test_news
    )[0]

    confidence = max(probability) * 100

    print("\n" + "=" * 60)
    print("TEST PREDICTION")
    print("=" * 60)

    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    train_system()