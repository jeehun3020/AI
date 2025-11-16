🚀 Transformer-Based Sentiment Analysis Project

This repository contains my work for the Natural Language Processing (NLP) assignment focused on implementing Transformer-based sentiment analysis using BERT modules and a custom decoder architecture.

This project explores how modern Transformer architectures can be adapted for text understanding tasks and compares their performance with earlier RNN-based and CNN-based approaches.

📌 Project Overview

The goal of this project is to build a sentiment analysis model capable of classifying movie reviews from the IMDB dataset as positive or negative.

Key Features
	•	🔹 BERT Preprocessing + BERT Encoder from TensorFlow Hub
	•	🔹 Custom Transformer-based Decoder
	•	🔹 Fully end-to-end fine-tuning
	•	🔹 IMDB dataset (25,000 reviews)
	•	🔹 Achieved high accuracy (~0.83) on validation data

This project demonstrates how Transformer models outperform earlier architectures by capturing contextual meaning across entire sequences.

🧠 Model Architecture

The final model consists of:
	1.	Input Layer — raw text
	2.	BERT Preprocessing Layer
	3.	BERT Encoder (trainable)
	4.	Transformer-Based Decoder
	5.	Sentiment Classification Head

The model produces a binary prediction representing positive or negative sentiment.

🧪 Training Results
	•	Epochs: 8
	•	Training Accuracy: steadily improved each epoch
	•	Validation Accuracy: peaked around 82–83%
	•	Test Accuracy: 0.8274

This confirms the performance boost gained from using a Transformer-based pipeline.

📈 Future Improvements
	•	Experiment with larger BERT variants (e.g., BERT-large, RoBERTa)
	•	Utilize data augmentation for text
	•	Incorporate learning-rate schedulers or warmup strategies
	•	Compare against LSTM- or GRU-based baselines
