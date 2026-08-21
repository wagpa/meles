import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

from meles import logs, scoring

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)


def main(dataset_path: Path, output_path: Path, method: Dict):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the input path
    dataset_df = pd.read_parquet(dataset_path)

    metrics = []
    for fold in sorted(dataset_df["fold"].unique()):
        dataset_k_df = dataset_df[dataset_df["fold"] == fold]

        # Extract the embeddings and labels for each identity
        embeddings = np.vstack(dataset_k_df["embedding"].to_numpy()).astype(float)
        print(f"{embeddings.shape} with {len(dataset_k_df["embedding"])} rows")
        labels = dataset_k_df["identity"].to_numpy()
        n_identities = len(np.unique(labels))

        # Fit the KMeans model
        model = KMeans(n_clusters=n_identities, random_state=42)
        model_labels = model.fit_predict(embeddings)

        # Score the model
        print(f"{len(model_labels)} clusters identified for fold {fold} of {n_identities} identities: {len(embeddings)} embeddings")
        silhouette_avg = silhouette_score(embeddings, model_labels)
        ari = adjusted_rand_score(labels, model_labels)

        metrics.append({
            "silhouette_avg": silhouette_avg,
            "adjusted_rand_score": ari,
            "fold": int(fold),
        })

    # Store the results, tagged with the wildcard combination that produced them. Without the tag
    # the rows are indistinguishable once 'collect' concatenates every point of the grid.
    pd.DataFrame(metrics).assign(**method).to_parquet(output_path)


if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        dataset_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        method={
            "frame_anonymization_method": snakemake.wildcards.get("frame_anonymization"),
            "frame_embedding_method": snakemake.wildcards.get("frame_embedding"),
            "video_aggregation_method": snakemake.wildcards.get("video_aggregation"),
            "video_embedding_method": snakemake.wildcards.get("video_embedding"),
            "identity_aggregation_method": snakemake.wildcards.get("identity_aggregation"),
        }
    )
