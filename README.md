# Text Generation Model

This repository contains a script to train and generate text using an LSTM neural network.

## Requirements

- Python 3.x
- TensorFlow
- Numpy
- Pandas
- NLTK

## Setup

1. Install the required packages:

```sh

pip install -r requirements.txt

#Example Usage:

python
Copy code
from text_generation import generate_text, load_model_and_history

model, history = load_model_and_history()
print(generate_text(model, "I will have to look into this thing because I", unique_token_index, unique_tokens, 10, 100, 10))
print(generate_text(model, "The president of the United States announced yesterday that he", unique_token_index, unique_tokens, 10, 100, 10))
