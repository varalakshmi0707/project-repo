import pandas as pd


def clean_hiring_data(
    input_file: str = "hiring_dataset.csv",
    output_file: str = "cleaned_data.csv",
) -> None:
    # Load CSV file
    df = pd.read_csv(input_file)

    # Fill missing numeric values with the median of each numeric column
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing text values with "Unknown"
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].fillna("Unknown")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")


if __name__ == "__main__":
    clean_hiring_data()
