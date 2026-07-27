import click
from pathlib import Path
import logging
import pandas as pd

from meles.aggregation.aggregator import NoAggregator
from meles.aggregation import (pooling, clustering, quality)

logger = logging.getLogger(__name__)

@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--train-output", "train_output_path", required=True, type=click.Path(path_type=Path))
@click.option("--test-output", "test_output_path", required=True, type=click.Path(path_type=Path))
@click.option("--method", required=True)

def main(input_path: Path, train_output_path: Path, test_output_path: Path, method: str) -> None:
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run the anonymizer for each frame
    logger.debug("Reading input file")
    structure = pd.read_parquet(input_path)
    train_structure = structure[structure["train"] == True]
    test_structure = structure[structure["train"] == False]

    # Select the aggregator for the selected method.
    agg = NoAggregator()
    match method:
        case "none":
            pass
        case "pooling_mean":
            agg = pooling.MeanPoolingAggregator()
        case "pooling_min":
            agg = pooling.MinPoolingAggregator()
        case "pooling_max":
            agg = pooling.MaxPoolingAggregator()
        case "pooling_median":
            agg = pooling.MedianPoolingAggregator()
        case "clustering_mean_d1_min3":
            agg = clustering.MeanClusteringAggregator(
                distance_threshold=1.0,
                min_cluster_size=3,
            )
        case "clustering_median_d1_min3":
            agg = clustering.MedianClusteringAggregator(
                distance_threshold=1.0,
                min_cluster_size=3,
            )
        case "clustering_mean_d08_min3":
            agg = clustering.MeanClusteringAggregator(
                distance_threshold=0.8,
                min_cluster_size=3,
            )
        case "clustering_median_d08_min3":
            agg = clustering.MedianClusteringAggregator(
                distance_threshold=0.8,
                min_cluster_size=3,
            )
        case "clustering_mean_d1_min3":
            agg = clustering.MeanClusteringAggregator(
                distance_threshold=1.0,
                min_cluster_size=3,
            )
        case "clustering_mean_d1_min3":
            agg = clustering.MeanClusteringAggregator(
                distance_threshold=1.0,
                min_cluster_size=3,
            )
        case "clustering_mean_d08_min3":
            agg = clustering.MeanClusteringAggregator(
                distance_threshold=0.8,
                min_cluster_size=3,
            )
        case "clustering_mean_d08_min3":
            agg = clustering.MeanClusteringAggregator(
                distance_threshold=0.8,
                min_cluster_size=3,
            )
        case _:
            raise ValueError(f"Unknown method: {method}")

    # Aggregate the train embeddings per identity (multiple videos)
    train_embeddings = (
        train_structure.groupby(['identity'])['embedding']
        .apply(lambda embeddings: agg.aggregate(embeddings.to_list()))
        .explode()
        .rename('embedding')
        .reset_index()
        .dropna(subset=['embedding'])
    )

    # Aggregate the test embeddings per identity-video (one video)
    test_embeddings = (
        test_structure.groupby(['identity', 'video'])['embedding']
        .apply(lambda embeddings: agg.aggregate(embeddings.to_list()))
        .explode()
        .rename('embedding')
        .reset_index()
        .dropna(subset=['embedding'])
    )

    train_embeddings.to_parquet(train_output_path, index=False)
    test_embeddings.to_parquet(test_output_path, index=False)

if __name__ == "__main__":
    main()