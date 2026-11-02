from ml_cardio.data.data_loader import load_dataset
from ml_cardio.preprocessing.prep import clean_data
from ml_cardio.preprocessing.split import split_data, save_splits


def main():
    raw = load_dataset()
    df = clean_data(raw)
    print(f"Rows: {len(raw)} -> {len(df)} after cleaning")

    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(df, X_train, X_test, y_train, y_test)

    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Positive rate – train:", round(y_train.mean(), 3), "| test:", round(y_test.mean(), 3))


if __name__ == "__main__":
    main()