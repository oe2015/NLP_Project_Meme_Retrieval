# NLP Project: Meme Retrieval with CLIP and Siglip

This project demonstrates how to train and validate CLIP and Siglip models for meme retrieval tasks using both literal and contextual captions. The models align image and text embeddings to enable effective retrieval based on natural language queries.

---

## Prerequisites

Before running the project, ensure you have the following:
- Python 3.8 or later
- A GPU-enabled environment for faster training and inference (optional but recommended)

---

## Setup Instructions

### 1. Download the Dataset
The dataset (the `images` directory) must be downloaded from the following [link](https://drive.google.com/drive/folders/1WczSeTOI6y_KzT5LeG36f7hWPawtV8Nz?usp=sharing). After downloading:
- Extract and place the `images` directory in the project root.

### 2. Install Dependencies
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. How to Run
To train and validate the CLIP model:

```bash
python clip_trainval.py
```

To train and validate the Siglip model:

```bash
python siglip_trainval.py
```

### 4. Notes

    The checkpoint_path in both scripts must be updated with the correct path to save or load model checkpoints.
    Adjust the batch_size and learning_rate in the scripts based on your hardware capabilities.

### 5. Contributing

If you'd like to contribute to this project, please fork the repository and create a pull request.
