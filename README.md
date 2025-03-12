# Neural Machine Translation (NMT) - Spanish to English

## Overview
This project involves building an **attention-based sequence-to-sequence** model for **Neural Machine Translation (NMT)** to translate Spanish sentences into English. The model is designed to help a **US-based life insurance company** communicate with the Spanish-speaking community in Mexico by translating application request letters.

## Problem Statement
Due to language barriers, Spanish-speaking individuals in Mexico face challenges understanding English-based communication from the insurance company. The company needs an **automated translation model** to bridge this gap. This project aims to develop a **context-aware** and **coherent** Spanish-to-English NMT model.

## Dataset
The dataset consists of **paired Spanish-English sentences**, provided by **Anki**. It contains parallel text corpora essential for training the translation model.

### Dataset Download Link
[Anki Parallel Corpus](https://www.manythings.org/anki/)

## Project Pipeline

### 1. Data Understanding & Preprocessing
- Load the dataset and explore sentence structures.
- Clean the dataset by removing special characters, handling accents, and tokenizing words.
- Create input-output pairs for model training.

### 2. Sequence-to-Sequence Model with Attention
- Implement an **Encoder-Decoder** model with **Bahdanau Attention**.
- Use **GRU/LSTM** for sequence processing.
- Apply **word embeddings** for meaningful representation.

### 3. Model Training
- Use an appropriate **optimizer** and **loss function**.
- Train the model for multiple **epochs** until convergence.
- Evaluate model performance using **BLEU Score** and other NLP metrics.

### 4. Testing & Inference
- Translate unseen Spanish sentences into English.
- Assess translation quality through human evaluation.
- Compare with traditional translation methods.

## Technologies Used
- **TensorFlow/Keras**: Model development
- **NLTK/Spacy**: Text preprocessing
- **Matplotlib/Seaborn**: Data visualization
- **BLEU Score**: Translation evaluation

## Setup Instructions
1. Install dependencies:
   ```sh
   pip install tensorflow keras nltk spacy matplotlib seaborn
   ```
2. Download and preprocess the dataset.
3. Train the model using the provided stub code.
4. Test the model with new Spanish sentences.

## Findings & Observations
- Challenges in handling **special characters** in Spanish.
- Importance of **attention mechanisms** in improving translation quality.
- Comparison between **BLEU score** and human evaluation.

