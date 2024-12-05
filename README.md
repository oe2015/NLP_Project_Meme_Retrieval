Here’s the updated README file with a properly functioning table of contents and incorporating details about the preprocessing:

---

# Fine-Tuning Sentence Transformer for Meme Retrieval

This repository contains a Jupyter Notebook that demonstrates the fine-tuning of a Sentence Transformer model for a meme retrieval task. The notebook preprocesses the MemeCap dataset with added OCR text, creates a triplet dataset for training, and evaluates the performance of the Sentence Transformer model using various loss functions.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Preprocessing](#preprocessing)
4. [Requirements](#requirements)
5. [Workflow](#workflow)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributing](#contributing)

---

## Introduction

This project is designed to improve meme retrieval by fine-tuning a Sentence Transformer model. The goal is to align meme contextual captions with the corresponding OCR-extracted text of the meme.

The task focuses on scenarios where users provide vague queries to search for meme images, aiming to retrieve the most contextually relevant memes.

---

## Dataset Overview

The dataset used in this project is the memeCap dataset, which includes annotations for memes such as:

- **Meme Captions**: Textual captions describing the context or humor of the memes.

Additionally, a new **OCR-Extracted** Text column was added by performing Optical Character Recognition (OCR) on the meme images. This extra column enriches the dataset by providing textual content directly extracted from the images, offering a complementary perspective to the contextual captions.

---


## Preprocessing

Preprocessing involves cleaning and normalizing the captions and OCR-extracted text. The following transformations are applied:

1. **Removing Special Tokens**: `<s>` and similar tokens are removed.
2. **Special Character Removal**: Non-alphanumeric characters are stripped.
3. **Case Normalization**: Text is converted to lowercase.
4. **Tokenization and Stopword Removal**: Text is tokenized, and common stopwords are filtered out.


---

## Requirements

All required libraries are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `sentence-transformers`
- `torch`
- `datasets`
- `nltk`

---

## Workflow

1. **Preprocessing**:
   - Clean and normalize contextual captions and OCR text.

2. **Creating Triplet Dataset**:
   - Generate (anchor, positive, negative) triplets for contrastive learning.

3. **Base Model Evaluation**:
   - Evaluate the pretrained Sentence Transformer on the meme retrieval task using contextual captions and OCR text.

4. **Fine-Tuning**:
   - Fine-tune the model with:
     - **Triplet Loss**: Improves understanding of relative similarity.
     - **Multiple Negative Ranking Loss**: Optimizes ranking across batches.

5. **Evaluation**:
   - Compare retrieval performance before and after fine-tuning.

---

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/oe2015/Wheres-that-Meme.git
    cd Wheres-that-Meme
    git branch sentence_transformer
    ```

2. Launch the notebook:
    ```bash
    jupyter notebook sentence_transformer.ipynb
    ```

3. Follow the step-by-step instructions to preprocess the dataset, fine-tune the model, and evaluate its performance.

---

## Results

The notebook evaluates:
- Baseline performance of the pretrained Sentence Transformer.
- Improvements after fine-tuning with:
  - **Triplet Loss**
  - **Multiple Negative Ranking Loss**

Evaluation includes Recall@k and MRR metrics, demonstrating performance enhancements.

---

## Contributing

Contributions are welcome! Please feel free to:
- Open issues for bugs or feature requests.
- Submit pull requests with improvements or new features.

