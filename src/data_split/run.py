import argparse
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

def go(args):

    run = wandb.init(job_type="data_split")

    # Download input artifact
    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    # Stratify if needed
    stratify_col = df[args.stratify_by] if args.stratify_by != "none" else None

    # Train / temp split
    train_df, temp_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_col
    )

    # Validation from temp
    stratify_temp = temp_df[args.stratify_by] if args.stratify_by != "none" else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=args.val_size / (1 - args.test_size),
        random_state=args.random_seed,
        stratify=stratify_temp
    )

    # Save locally
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    # Log artifacts
    for name, file in [
        ("train.csv", "train_data"),
        ("val.csv", "val_data"),
        ("test.csv", "test_data"),
    ]:
        artifact = wandb.Artifact(name, type=file)
        artifact.add_file(name)
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset")

    parser.add_argument("--input_artifact", type=str, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)

    args = parser.parse_args()

    go(args)
