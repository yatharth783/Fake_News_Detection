import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from preprocessing import clean_text


def train_system():

    # =========================
    # 1. CREATE FOLDERS
    # =========================

    os.makedirs("models", exist_ok=True)
    os.makedirs("notebooks", exist_ok=True)

    # =========================
    # 2. FILE PATHS
    # =========================

    true_path = "data/True.csv"
    fake_path = "data/Fake.csv"

    print("Loading dataset...")

    # IMPORTANT:
    # Dataset is TAB separated
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

    # =========================
    # 3. CLEAN COLUMN NAMES
    # =========================

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

    # =========================
    # 4. ADD LABELS
    # =========================

    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    # =========================
    # 5. COMBINE DATA
    # =========================

    df = pd.concat(
        [true_df, fake_df],
        ignore_index=True
    )

    print("\nDataset Size:", df.shape)

    # =========================
    # 6. CREATE CONTENT
    # =========================

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

    # Remove empty articles
    df = df[
        df["content"].str.strip() != ""
    ]

    # =========================
    # 7. CLASS DISTRIBUTION
    # =========================

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

    # =========================
    # 8. PREPROCESSING
    # =========================

    print(
        "\nPreprocessing text..."
    )

    df["cleaned_content"] = (
        df["content"].apply(clean_text)
    )

    X = df["cleaned_content"]
    y = df["label"]

    # =========================
    # 9. TRAIN TEST SPLIT
    # =========================

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

    # =========================
    # 10. MODELS
    # =========================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),

        "Naive Bayes":
            MultinomialNB(),

        "Linear SVM":
            LinearSVC()
    }

    best_model = None
    best_score = 0
    best_name = ""

    # =========================
    # 11. TRAIN MODELS
    # =========================

    for name, classifier in models.items():

        print("\n" + "=" * 50)
        print("Model:", name)
        print("=" * 50)

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

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(
            X_test
        )

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        score = f1_score(
            y_test,
            y_pred,
            pos_label="FAKE"
        )

        print(
            "F1 Score:",
            round(score, 4)
        )

        if score > best_score:

            best_score = score
            best_model = pipeline
            best_name = name

    # =========================
    # 12. SAVE MODEL
    # =========================

    print("\n" + "=" * 50)

    print(
        f"Best Model: {best_name}"
    )

    print(
        f"Best F1-Score: {best_score:.4f}"
    )

    print("=" * 50)

    joblib.dump(
        best_model,
        "models/fake_news_model.pkl"
    )

    print(
        "\nModel saved successfully!"
    )

    print(
        "models/fake_news_model.pkl"
    )


if __name__ == "__main__":
    train_system()