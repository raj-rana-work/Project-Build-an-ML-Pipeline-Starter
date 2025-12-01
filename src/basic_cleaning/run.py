#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):

    # Start W&B run
    run = wandb.init(
        project="nyc_airbnb",
        group="cleaning",
        job_type="basic_cleaning",
        save_code=True
    )
    run.config.update(args)

    # Download input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # Drop rows where last_review is missing (matches reference behavior)
    df = df.dropna(subset=["last_review"])

    # Keep only listings inside NYC boundaries
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Filter out listings with zero availability (matches reference dataset)
    df = df[df['availability_365'] > 0].copy()

    # Drop rows with missing host_name (reference does not contain these)
    df = df.dropna(subset=['host_name'])

    # NEW: match reference dataset (only entire home/apartment)
    df = df[df['room_type'] == "Entire home/apt"].copy()

    # Apply price filter again (ensure consistency after filtering)
    df = df[df['price'].between(min_price, max_price)].copy()

    # Save the cleaned data
    df.to_csv("clean_sample.csv", index=False)

    # Log the cleaned dataset as an artifact
    # IMPORTANT: artifact name MUST NOT include .csv
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()


# Argument definitions
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact in W&B, e.g. 'sample.csv:latest'",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output cleaned artifact to create (NO .csv), e.g. 'clean_sample'",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type/category of the output artifact, e.g. 'clean_sample'",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A short description of what the cleaned artifact represents",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum allowed price; rows with price < min_price will be removed",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum allowed price; rows with price > max_price will be removed",
        required=True,
    )

    args = parser.parse_args()

    go(args)
