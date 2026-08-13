import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING, Dict

from meles import scoring

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

def main(train_path: Path, test_path: Path, split_path: Path, output_path: Path, method: Dict):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the input paths
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    split_df = pd.read_parquet(split_path)

    # Add the split information to the test data.
    test_df = test_df.merge(split_df, on=['identity', 'video'], how='left')

    # Score each fold separately
    metrics = []
    for k in range(split_df["fold"].max()):
        # Filter the dataframes to only include the rows with the correct folds.
        # For each fold, we use the matching index as the train set and the rest as test sets.
        train_k_df = train_df[train_df["fold"] == (k + 1)]
        test_k_df = test_df[test_df["fold"] != (k + 1)]

        # Collect the embeddings. They match the 'y_train' labels by index.
        # Then, fit the classifier.
        x_train = scoring.stack(train_k_df)
        y_train = train_k_df["identity"].to_numpy()
        classifier = scoring.fit_gallery(x_train, y_train)

        # Now generate the scoring matrix for each test row.
        x_test = scoring.stack(test_k_df)
        y_test = test_k_df["identity"].to_numpy()
        n_neighbors = min(scoring.K, len(train_k_df))
        classes = scoring.label_space(y_train, y_test)
        score_matrix = scoring.vote_matrix(
            classifier,
            embeddings=x_test,
            classes=classes,
            n_neighbors=n_neighbors
        )

        # Lastly, generate the accuracy scores
        keys = scoring.video_keys(test_k_df)
        video_scores, y_true = scoring.pool_per_video(score_matrix, keys)
        metrics_k = scoring.top_k_accuracies(y_true, video_scores, classes)
        metrics_k["frame-top-1"] = scoring.row_accuracy(score_matrix, y_test, classes)
        metrics.append(metrics_k)

    # Store the results
    pd.DataFrame(metrics).to_parquet(output_path)

if __name__ == "__main__":
    # noinspection unbound-local-variable
    main(
        train_path=snakemake.input.train_dataset,
        test_path=snakemake.input.test_dataset,
        split_path=snakemake.input.k_split,
        output_path=snakemake.output.dataset,
        method={
            "frame_anonymization_method": snakemake.wildcards.get("frame_anonymization"),
            "frame_embedding_method": snakemake.wildcards.get("frame_embedding"),
            "video_aggregation_method": snakemake.wildcards.get("video_aggregation"),
            "video_embedding_method": snakemake.wildcards.get("video_embedding"),
            "identity_aggregation_method": snakemake.wildcards.get("identity_aggregation"),
        }
    )
