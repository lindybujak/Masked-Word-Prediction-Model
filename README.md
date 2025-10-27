# Masked-Word-Prediction-Model
A PyTorch implementation of a BERT-style masked language model built from scratch. We designed custom tokenization, datasets, and transformer architectures to predict masked words and explore how attention learns word context.

Created by Lindy Bujak & Melia Pan

This project is our implementation of a Masked Word Prediction Model, inspired by BERT, completed as part of our coursework for DATASCI 315: Machine Learning in Python. This was developed for a Kaggle competition for our class, which our team placed 1st in terms of accuracy on unseen data. This repository represents our personal, extended version where we experimented with different tokenization strategies, model architectures, and data preprocessing techniques beyond the assignment requirements.

The goal of this challenge was to predict masked words in sentences using a small artificial dataset of 20 unique words. The training set consisted of 10,000 complete sentences, while the test set contained 30,000 sentences with one <mask> token per sentence. Despite the simplicity of the language, this task provided an excellent opportunity to explore transformer-based architectures and understand how attention mechanisms can model word relationships.

In this notebook, we:
* Implemented a custom tokenizer and vocabulary builder from scratch
* Designed a custom PyTorch Dataset that randomly masks tokens during training
* Built and trained multiple transformer-based architectures, including:
  * A basic Transformer Encoder model
  * An improved architecture with GELU activation, layer normalization, and dropout
  * A BERT-inspired transformer with learnable positional embeddings and a masked language modeling head
* Implemented training, validation, and evaluation loops to measure accuracy on masked tokens
* Generated predictions on the test set and formatted them for Kaggle submission

Our final model achieved strong accuracy and provided meaningful insights into how transformers learn contextual relationships even from limited data. The project also served as a hands-on exercise in model design, optimization, and reproducible experimentation in PyTorch.
