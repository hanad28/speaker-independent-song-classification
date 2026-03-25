# Speaker-Independent Song Classification from Hummed and Whistled Audio

An end-to-end audio machine learning pipeline that predicts one of eight song labels from 10-second hummed or whistled recordings, using participant-disjoint evaluation to test generalisation to completely unseen users.

## Overview

This project explores a challenging multiclass audio-classification task: identifying the song being performed in a short humming or whistling recording.

The system takes a raw 10-second audio clip as input, converts it into a compact hand-crafted feature representation, and applies classical machine learning models to predict one of eight song classes. A key design choice is the use of **participant-disjoint train, validation, and test splits**, ensuring that no individual appears in more than one subset. This makes the evaluation more realistic by testing whether the model generalises to unseen users rather than memorising speaker-specific traits.

Rather than chasing headline accuracy alone, this project focuses on **sound methodology, leakage prevention, feature engineering, model comparison, and honest evaluation under difficult real-world conditions**.

## Problem Statement

Melody recognition from humming and whistling is difficult for several reasons:

- raw audio is high dimensional
- people hum and whistle differently
- recordings vary in timing, pitch stability, loudness, and background noise
- some song fragments are acoustically similar

This project investigates how far a carefully designed **classical ML pipeline** can go on this task before richer learned audio representations become necessary.

## Dataset

The project uses the **MLEnd Hums and Whistles II** dataset, a collection of short audio recordings contributed by students as humming or whistling interpretations of movie-song fragments.

### Dataset characteristics

- **800 total recordings**
- **8 song classes**
- **100 recordings per class**
- **400 hum recordings**
- **400 whistle recordings**
- **187 unique participants**

Each file includes metadata such as:

- song label
- interpretation type (`hum` or `whistle`)
- interpreter ID
- recording number

This creates a balanced multiclass problem while still preserving substantial real-world variation across users and recording styles.

## Objective

Given a 10-second audio clip, predict which of the following eight songs it represents:

- Despicable Me (Happy): https://youtu.be/MOWDb2TBYDg?t=28
- Zootropolis (TryEverything): https://youtu.be/c6rP-YP4c5I?list=RDc6rP-YP4c5I&t=65 
- Coco (RememberMe): https://youtu.be/KP_XkN2v7OM?list=RDKP_XkN2v7OM&t=26 
- Madagascar (NewYork): https://www.youtube.com/watch?v=le1QF3uoQNg&t=16s 
- Toy Story (Friend): https://youtu.be/0hG-2tQtdlE?list=RD0hG-2tQtdlE&t=10 
- Jungle Book (Necessities): https://youtu.be/6BH-Rxd-NBo 
- Trolls (Feeling): https://youtu.be/oWgTqLCLE8k?t=45 
- Up (Married): https://youtu.be/S1uWHjhluTI?t=10

## Project Pipeline

The overall workflow is:

- **Raw audio**
- **Normalisation**
- **Feature extraction**
- **Feature scaling**
- **Model training and validation**
- **Final model selection**
- **Held-out test evaluation**

## Preprocessing

Each audio clip is normalised to a consistent format before feature extraction:

- resampled to **16 kHz**
- converted to **mono**
- truncated or zero-padded to exactly **10 seconds**
- represented as a waveform of **160,000 samples**

This ensures all clips have the same temporal structure and makes downstream feature extraction comparable across recordings.

## Feature Engineering

Raw waveforms are too high dimensional for reliable classical ML on a dataset of this size, so each clip is transformed into a **50-dimensional feature vector**.

### Feature families used

#### 1. MFCC statistics
Captures broad timbral and spectral-envelope information.

- mean and standard deviation of 13 MFCC coefficients

#### 2. Spectral features
Captures frequency distribution and timbral structure.

- spectral centroid
- spectral bandwidth
- spectral roll-off
- spectral contrast
- zero-crossing rate

#### 3. Pitch-based features
Captures melodic contour and vocal stability.

- pitch mean
- pitch variance
- pitch range
- voiced-frame fraction
- pitch jumps

#### 4. Energy and temporal features
Captures loudness and how energy evolves over time.

- RMS mean and standard deviation
- signal power
- temporal centroid
- attack time

Together, these features provide a compact summary of timbre, pitch, frequency structure, and dynamics.

## Experimental Design

A key methodological decision in this project is the use of **group-aware splitting by participant ID**.

### Why this matters

If recordings from the same participant appeared in both training and evaluation sets, the model could learn person-specific vocal habits rather than song-specific structure. To avoid this, the dataset is split so that:

- training participants are unique to training
- validation participants are unique to validation
- test participants are unique to test

This creates a much stricter and more realistic evaluation setting.

### Evaluation approach

Models were assessed using:

- training accuracy
- validation accuracy
- confusion matrices
- precision
- recall
- F1-score

The final test set was used **once only**, after model selection, to preserve an unbiased estimate of deployment performance.

## Models Compared

A range of classical ML models was explored to test different inductive biases and complexity levels.

### Baselines
- Majority-class classifier
- k-Nearest Neighbours

### Linear models
- Logistic Regression
- Linear SVM

### Non-linear models
- RBF SVM
- Multilayer Perceptron

### Tree-based models
- Random Forest
- Histogram Gradient Boosting

### Ensemble
- Soft-voting ensemble combining:
  - Linear SVM
  - Logistic Regression
  - Histogram Gradient Boosting

## Model Selection

The **soft-voting ensemble** achieved the highest validation accuracy:

- **Soft-voting ensemble:** 0.2970
- **Linear SVM:** 0.2909

However, the improvement was **less than one percentage point**, so the final model selected for retraining and test evaluation was the **Linear SVM**.

### Why Linear SVM was chosen

- strong validation performance
- simpler and more reproducible than the ensemble
- lower risk of overfitting than more flexible models
- cleaner trade-off between performance and complexity

This reflects a deliberate modelling decision: choosing the most defensible and robust pipeline rather than the most complicated one.

## Key Findings

- Classical ML can extract some useful structure from hummed and whistled melody recordings, but the task remains difficult.
- Careful preprocessing and engineered audio features are essential because raw waveforms are too large and noisy for this setting.
- Participant-disjoint evaluation is crucial; without it, performance would likely be overly optimistic.
- Linear models, especially **Linear SVM**, were more reliable than several more flexible alternatives.
- The main bottleneck appears to be the **representational power of the hand-crafted features**, not simply classifier choice.

## Limitations

This project is intentionally honest about its limits.

- The task itself is intrinsically hard.
- Recordings vary heavily in articulation, pitch control, rhythm, loudness, and background noise.
- The dataset is balanced but still relatively small for audio classification.
- Group-aware splitting makes the task more realistic, but also harder because the model must generalise across unseen speakers.
- Hand-crafted features are informative, but they likely lose some structure present in the raw waveform.

## Future Improvements

Several extensions could improve performance and make the system more production-ready:

- use mel-spectrograms or learned embeddings instead of only hand-crafted features
- compare against convolutional or transformer-based audio models
- perform cross-validation with grouped folds
- analyse class confusions in more detail to identify musically similar song pairs
- test separate pipelines for humming and whistling
- add audio augmentation to improve robustness
- package the pipeline into a reusable prediction script or lightweight demo app

## Tech Stack

- Python
- NumPy
- pandas
- matplotlib
- librosa
- scikit-learn
- Jupyter Notebook

## Why This Project Matters

This project demonstrates more than just training a classifier. It shows the ability to:

- design an end-to-end ML pipeline
- engineer meaningful features from raw audio
- prevent leakage through careful experimental design
- compare multiple model families rigorously
- make defensible trade-offs between accuracy and complexity
- communicate limitations honestly

That combination is exactly what makes a machine learning project credible in a portfolio.

## Acknowledgements

This project uses the MLEnd Hums and Whistles II dataset and was originally developed as part of the *Principles of Machine Learning* mini-project.

## How to Run

1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. Obtain access to the MLEnd Hums and Whistles II dataset through the appropriate authorised source.
4. Place the dataset files in the `data/` directory.
5. Open `notebooks/song_classification_pipeline.ipynb`.
6. Run the notebook cells in order to reproduce preprocessing, feature extraction, model training, and evaluation.

## Repository Structure

```text
speaker-independent-song-classification/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── song_classification_pipeline.ipynb
├── .gitignore
└── LICENSE


