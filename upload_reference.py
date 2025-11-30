import wandb

run = wandb.init(
    project="nyc_airbnb",
    job_type="upload_reference",
    reinit=True  # <<< IMPORTANT
)

# Unique artifact name (NOT clean_sample.csv)
artifact = wandb.Artifact(
    name="clean_sample_reference",
    type="reference",
    description="Reference dataset for data validation"
)

artifact.add_file("clean_sample.csv")

# Alias “reference” is CRITICAL – tests look for this alias
run.log_artifact(artifact, aliases=["reference"])

run.finish()
