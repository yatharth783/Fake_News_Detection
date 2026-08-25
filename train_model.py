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

from preprocessing import clean_text


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
    # 3. LOAD DATASET
    # =====================================================

    true_df = pd.read_csv(
        true_path,
        sep="\t",
        engine="python",
        on_bad_lines="skip"
    )

    fake_df = pd.read_csv(
        fake_path,
        sep="\t",
        engine="python",
        on_bad_lines="skip"
    )


    # =====================================================
    # 4. CLEAN COLUMN NAMES
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
    # 5. ADD LABELS
    # =====================================================

    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"


    # =====================================================
    # 6. COMBINE DATA
    # =====================================================

    df = pd.concat(
        [true_df, fake_df],
        ignore_index=True
    )

    print("\nDataset Size:", df.shape)


    # =====================================================
    # 7. CREATE CONTENT
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

    elif "title" in df.columns:

        df["content"] = (
            df["title"]
            .fillna("")
            .astype(str)
        )

    else:

        print("\nERROR: title/text columns not found.")

        print(
            "Available columns:",
            df.columns.tolist()
        )

        return


    # =====================================================
    # 8. REMOVE EMPTY ARTICLES
    # =====================================================

    df = df[
        df["content"].str.strip() != ""
    ]

    print("\nFinal Dataset Size:", df.shape)


    # =====================================================
    # 9. CLASS DISTRIBUTION
    # =====================================================

    print("\nClass Distribution:")

    print(
        df["label"].value_counts()
    )

    df["label"].value_counts().plot(
        kind="bar"
    )

    plt.title(
        "Real vs Fake News Distribution"
    )

    plt.xlabel("News Type")
    plt.ylabel("Number of Articles")

    plt.tight_layout()

    plt.savefig(
        "notebooks/distribution.png"
    )

    plt.close()


    # =====================================================
    # 10. PREPROCESSING
    # =====================================================

    print("\nPreprocessing text...")

    df["cleaned_content"] = (
        df["content"].apply(clean_text)
    )

    X = df["cleaned_content"]
    y = df["label"]


    # =====================================================
    # 11. TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
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
    # 12. MODELS
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
    # 13. BEST MODEL VARIABLES
    # =====================================================

    best_model = None
    best_score = 0
    best_name = ""

    best_metrics = {}


    # =====================================================
    # 14. TRAIN MODELS
    # =====================================================

    for name, classifier in models.items():

        print("\n" + "=" * 60)
        print("Model:", name)
        print("=" * 60)


        # -------------------------------------------------
        # Pipeline
        # -------------------------------------------------

        pipeline = Pipeline([

            (
                "tfidf",

                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )
            ),

            (
                "classifier",
                classifier
            )
        ])


        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        y_pred = pipeline.predict(
            X_test
        )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Classification Report
        # -------------------------------------------------

        print(
            classification_report(
                y_test,
                y_pred
            )
        )


        # -------------------------------------------------
        # Print Metrics
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Select Best Model
        # -------------------------------------------------

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
    # 15. FINAL RESULT
    # =====================================================

    print("\n" + "=" * 60)

    print(
        f"🏆 Best Model: {best_name}"
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
    # 16. SAVE BEST MODEL
    # =====================================================

    model_path = (
        "models/fake_news_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    print(
        "\n✅ Model saved successfully!"
    )

    print(
        f"📁 {model_path}"
    )


# =========================================================
# RUN TRAINING
# =========================================================

if __name__ == "__main__":
    train_system()