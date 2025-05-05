# LSTM Sentiment Classifier

This project implements a Long Short-Term Memory (LSTM) based model for sentiment analysis. The model is trained to classify text data into binary sentiment categories (e.g., positive or negative) using deep learning techniques.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Overview

The goal of this project is to build an LSTM-based neural network that can accurately predict the sentiment of a given text input. The model leverages word embeddings and sequential modeling to understand the context and sentiment conveyed in the text.

## Dataset

The dataset used for training and evaluation consists of labeled text samples. Each sample includes a piece of text and a corresponding sentiment label:

- **Text**: The input sentence or document.
- **Label**: Binary sentiment label (0 for negative, 1 for positive).

*Note: Replace this section with specific details about your dataset, including the source, size, and any preprocessing steps applied.*

## Preprocessing

Before training the model, the text data undergoes several preprocessing steps:

1. **Lowercasing**: Converting all text to lowercase to maintain consistency.
2. **Punctuation Removal**: Eliminating punctuation marks to reduce noise.
3. **Stopword Removal**: Removing common stopwords that may not contribute to sentiment.
4. **Tokenization**: Splitting text into individual words or tokens.
5. **Padding**: Ensuring all sequences have the same length by padding shorter sequences.

These steps help in standardizing the input data and improving the model's performance.

## Model Architecture

The LSTM model is constructed using the following layers:

- **Embedding Layer**: Transforms each word into a fixed-length vector representation.
- **LSTM Layer**: Captures sequential dependencies and contextual information from the text.
- **Dropout Layer**: Prevents overfitting by randomly setting input units to 0 during training.
- **Dense Layer**: Fully connected layer that outputs the final prediction.

The model is compiled with a binary cross-entropy loss function and optimized using the Adam optimizer.

## Training

The model is trained on the preprocessed dataset with the following parameters:

- **Epochs**: 5
- **Batch Size**: 32
- **Validation Split**: 20% of the data is used for validation

During training, the model's accuracy and loss are monitored to assess its learning progress.

## Evaluation

After training, the model's performance is evaluated using metrics such as:

- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were identified correctly.
- **F1 Score**: Harmonic mean of precision and recall.

A confusion matrix is also generated to visualize the model's performance across different classes.

## Usage

To use the trained model for sentiment prediction on new text data:

1. **Preprocess** the input text using the same preprocessing steps as the training data.
2. **Tokenize** and **pad** the text to match the input format expected by the model.
3. **Predict** the sentiment using the model's `predict` function.
4. **Interpret** the output: if the predicted probability is greater than 0.5, the sentiment is positive; otherwise, it's negative.

### Example:

```python
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment

# Usage
text = "I absolutely loved the movie!"
print(predict_sentiment(text))
```

## Requirements

To run this project, ensure you have the following libraries installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- NLTK
- Matplotlib
- Seaborn

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

*Note: Create a `requirements.txt` file listing all the dependencies.*

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
