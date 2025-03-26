# MachineLearning_Proy1
Curso de Inteligencia Artificial - Primer Proyecto: Machine Learning

### PyTorch Neural Network Project
A collection of neural network implementations using PyTorch, focusing on classic machine learning architectures and tasks.

##  Team Members

- María José Villafuerte (22129)
- Fabiola Contreras (22787)
- Diego Duarte (22075)

## Project Overview

This project implements several neural network models for different tasks, ranging from simple perceptrons to more complex architectures like convolutional neural networks and transformers.

### Models Implemented:

- **Perceptron Model**: A simple binary classifier
- **Regression Model**: For function approximation
- **Digit Classification Model**: For MNIST handwritten digit classification
- **Convolutional Neural Network**: Enhanced digit classification
- **Transformer-based Models**: Including attention mechanisms and GPT-like architecture

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/Fabiola-cc/MachineLearning_Proy1.git

# Navigate to the project directory
cd MachineLearning_Proy1

```

## Running the Models

The project includes an autograder that can be used to test the implementations:

```bash
python autograder.py

python autograder.py -q q1
python autograder.py -q q2
python autograder.py -q q3
```

To run specific models:

```bash
# Train the perceptron model
python PerceptronModel.py

# Train the regression model
python RegressionModel.py

# Train the digit classification model
python DigitClassification.py
```

## Model Descriptions

### Perceptron Model
A single-layer perceptron for binary classification with manual weight updates.

### Regression Model
A multi-layer neural network designed to approximate continuous functions, specifically trained to model the sine function over [-2π, 2π].

### Digit Classification Model
A feedforward neural network that classifies handwritten digits (0-9) using the MNIST dataset.

### Convolutional Neural Network
An enhanced digit classification model that utilizes convolutional layers to better capture spatial features in the digit images.

### Transformer-based Language Model
A character-level language model inspired by GPT architecture, implementing self-attention mechanisms for text generation.

## Performance Metrics

The models are evaluated based on the following criteria:

- **Perceptron**: Classification accuracy on linearly separable data
- **Regression**: Mean squared error (target < 0.02)
- **Digit Classification**: Test accuracy (target > 97%)
- **CNN**: Test accuracy (target > 80%)
- **Language Model**: Validation accuracy (target > 81%)

## Project Structure

```
├── autograder.py           # Testing framework
├── backend.py              # Data handling and visualization
├── PerceptronModel.py      # Perceptron implementation
├── RegressionModel.py      # Regression model implementation
├── DigitClassification.py  # Digit classification models
├── gpt_model.py            # Transformer architecture
├── chargpt.py              # Character-level language model
└── data/                   # Data directory
    └── mnist.npz           # MNIST dataset
```


