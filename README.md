# 📰 Headline Generation Models: LSTM vs. Transformers

This repository explores two cutting-edge approaches to headline generation using neural networks: **Long Short-Term Memory (LSTM)** and **Transformers**. Each approach leverages different strengths of deep learning to tackle the challenge of generating coherent and contextually relevant headlines.

<p align="center">
   <img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white" />
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

## Overview

### 🔄 LSTM Approach

**Long Short-Term Memory (LSTM)** networks are a specialized type of **Recurrent Neural Network (RNN)** designed to capture long-term dependencies in sequential data. They are known for their ability to remember information over long sequences and maintain context, which is crucial for tasks like text generation.

**Key Features of LSTMs:**
- **Memory Cells:** LSTMs include memory cells that store information across sequences, which helps in retaining past contexts.
- **Gating Mechanisms:** They utilize input, output, and forget gates to regulate the flow of information, effectively managing long-term dependencies.
- **Sequential Processing:** LSTMs process input data one step at a time, evolving their internal state based on new inputs.

**Advantages in Headline Generation:**
- **Contextual Awareness:** LSTMs excel at maintaining context over longer sequences, which is essential for generating headlines that are coherent and contextually relevant.
- **Temporal Relationships:** They are effective in scenarios where the order and timing of words are important, such as generating text where prior words influence the subsequent ones.

### 🔀 Transformer Approach

The **Transformer** model, introduced in the paper "Attention is All You Need," represents a significant advancement in sequence modeling. **Transformers leverage self-attention mechanisms to handle long-range dependencies and process sequences in parallel.**

**Key Features of Transformers:**
- **Self-Attention Mechanism:** This mechanism enables the model to weigh the relevance of different words in a sequence, regardless of their position, allowing for a more comprehensive understanding of context.
- **Positional Encoding:** Transformers incorporate positional information into the input embeddings to maintain the order of words.
- **Parallel Processing:** Unlike LSTMs, Transformers process entire sequences simultaneously, leading to more efficient training and faster development.

**Advantages in Headline Generation:**
- **Global Context Understanding:** Transformers can capture complex relationships between words across the entire sequence, leading to more nuanced and contextually accurate headlines.
- **Efficient Training:** The ability to process sequences in parallel reduces training times, making Transformers more efficient for large datasets and quicker iterations.

## Comparison of LSTM and Transformer Approaches

| Feature                        | LSTM                                               | Transformer                                      |
|--------------------------------|----------------------------------------------------|--------------------------------------------------|
| **Architecture**               | Sequential, uses gates and memory cells           | Parallel, uses self-attention mechanisms         |
| **Context Handling**           | Maintains long-term dependencies through memory    | Captures global context with self-attention      |
| **Training Efficiency**        | Slower due to sequential processing                | Faster due to parallel processing                |
| **Complexity**                 | Simpler in terms of architecture                   | More complex with multiple layers and attention mechanisms |
| **Use Case Suitability**        | Effective for tasks with strong temporal dependencies | Superior for tasks requiring understanding of complex relationships across the entire sequence |

By comparing these two approaches, this project highlights their respective strengths and trade-offs in the context of headline generation. Whether you are interested in the sequential memory capabilities of LSTMs or the advanced attention mechanisms of Transformers, this repository offers a comprehensive guide to implementing and evaluating both methods.

## More Details 📝

For comprehensive information about this project, check out this [Medium article](https://sanchezsanchezsergio418.medium.com/lstm-vs-transformers-a-comparative-study-in-sequence-generation-310375867131).

## 📂 Repository Structure

This repository is organized to provide clear and practical examples for implementing and evaluating both LSTM and Transformer-based headline generation models. The structure is designed to facilitate both hands-on experimentation and code reuse.

### 📓 Notebooks

- **`LSTM_Headline_Generator.ipynb`**: This Jupyter notebook provides a comprehensive walkthrough for implementing and training a headline generation model using the Long Short-Term Memory (LSTM) architecture. It includes detailed sections on:
  - **Data Preprocessing**: Preparing and cleaning the dataset for use with the LSTM model.
  - **Model Creation**: Building the LSTM model architecture tailored for headline generation.
  - **Training**: Instructions and code for training the model, including hyperparameter tuning and validation.
  - **Evaluation**: Techniques and metrics for assessing the performance and quality of generated headlines.

- **`Transformer_Headline_Generator.ipynb`**: This Jupyter notebook covers the implementation and training of a headline generation model using Transformer architecture. It features:
  - **Data Preparation**: Steps to preprocess and format the data for use with Transformer models.
  - **Model Design**: Building the Transformer model, including attention mechanisms and positional encodings.
  - **Training**: Guidelines for training the Transformer model, with a focus on efficiency and effectiveness.
  - **Evaluation**: Methods for evaluating the model’s performance and quality of generated headlines.

### 🛠 Wrapper Classes

- **`LSTMHeadlineGenerator.py`**: This Python class wraps the trained LSTM model, providing a user-friendly interface for generating headlines. It includes:
  - **Model Loading**: Methods for loading pre-trained LSTM models and associated weights.
  - **Text Generation**: Functions to generate coherent headlines from input prompts, with options for customization.

- **`TransformersHeadlineGenerator.py`**: This Python class encapsulates the trained Transformer model, simplifying the process of generating headlines. Features include:
  - **Model Integration**: Functions for loading and utilizing the Transformer model, including handling pre-trained weights.
  - **Text Generation**: Tools to generate headlines based on prompts, with options to adjust generation parameters and improve output quality.

By organizing the repository in this manner, users can easily navigate between practical implementations and reusable components, enabling effective exploration and comparison of LSTM and Transformer models for headline generation.

## 🚀 Getting Started

### ✅ Prerequisites

Make sure you have the following installed:

- Python 3.x 🐍
- Jupyter Notebook 📓
- Required libraries (detailed in `requirements.txt`)

Install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

### 🛠 Usage

1. **Training the Models**:
   - Open `LSTM_Headline_Generator.ipynb` to explore the LSTM model's data preprocessing, training, and evaluation process.
   - Open `Transformer_Headline_Generator.ipynb` to see the implementation and training of the Transformer model.

2. **Generating Headlines**:
   - After training, use the wrapper classes to generate headlines. These classes handle everything internally, making it easy to test the models.
   - Example usage:
     ```python
     num_words_to_generate = 10
     start_prompt = "Blockchain"
     
     # Initialize the headline generators
     lstm_model = LSTMHeadlineGenerator()
     transformer_model = TransformersHeadlineGenerator()

     # Generate headlines
     headline_lstm = lstm_model.generate_text_from_prompt(start_prompt, num_words_to_generate)
     headline_transformer = transformer_model.generate_text_from_prompt(start_prompt, num_words_to_generate)

     print("📰 LSTM Headline:", headline_lstm) -> 'Blockchain Technology And Its Impact On The Financial Industry And Opportunities'
     print("📰 Transformer Headline:", headline_transformer) ->  'blockchain technology in the manufacturing : opportunities and conservation'
     ```


### 🤝 Contributing
We welcome contributions! If you have ideas for improving the models, adding new features, or enhancing the documentation, feel free to fork the repository and submit a pull request. 🙌

### 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

### 🙏 Acknowledgments

Special thanks to the authors of the papers and libraries used in this project, including:

* **Attention is All You Need** - The original Transformer paper.
* **Hochreiter & Schmidhuber** - The original LSTM paper.

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture architectures.

## Visitors Count

<img width="auto" src="https://profile-counter.glitch.me/headline_generation_lstm_transformers/count.svg" />

## Please Share & Star the repository to keep me motivated.
  <a href = "https://github.com/sergio11/headline_generation_lstm_transformers/stargazers">
     <img src = "https://img.shields.io/github/stars/sergio11/headline_generation_lstm_transformers" />
  </a>

