# Data

This project uses the **MLEnd Hums and Whistles II** dataset.

The raw audio files are **not included** in this repository. The dataset contains short hummed and whistled recordings of eight song classes, along with participant-level metadata used for group-aware splitting and evaluation.

## Dataset summary

- 800 total recordings
- 8 song classes
- 100 recordings per class
- 400 hum recordings
- 400 whistle recordings
- 187 unique participants

## Why the data is not included

The dataset is not uploaded here because it was provided for coursework use and is subject to permission restrictions.

## Reproducing the project

To reproduce the notebook:

1. Obtain access to the MLEnd Hums and Whistles II dataset through the appropriate course.
2. Place the audio files and metadata in this `data/` directory.
3. Update local file paths in the notebook if needed.
4. Run `notebooks/song_classification_pipeline.ipynb` from start to finish.

## Notes

This repository focuses on the machine learning pipeline, feature engineering, model comparison, and evaluation methodology rather than redistributing the raw dataset.
