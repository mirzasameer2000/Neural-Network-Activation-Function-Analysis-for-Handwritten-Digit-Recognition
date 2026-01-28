# Neural Network Activation Function Analysis for Handwritten Digit Recognition

## 📌 Project Overview
- A neural network–based handwritten digit recognition system.
- Focuses on analyzing how different activation functions affect learning, convergence, and generalization.
- Evaluated using the MNIST benchmark dataset and multiple real-world handwritten digit datasets.
- Implemented using PyTorch with a fully connected Multi-Layer Perceptron (MLP).

---

## 🎯 Objectives
- Implement a neural network for handwritten digit classification.
- Compare activation functions:
  - Sigmoid
  - Tanh
  - ReLU
  - Mixed activation configurations
- Evaluate training convergence and accuracy.
- Test generalization on external handwritten and online digit datasets.
- Perform extended training experiments with increased epochs.
- Analyze reliability, bias, and real-world deployment risks.

---

## 🧠 Model Architecture
- **Model Type:** Fully Connected Multi-Layer Perceptron (MLP)
- **Input Size:** 784 (flattened 28×28 grayscale images)
- **Hidden Layers:** 6 fully connected hidden layers
- **Activation Functions Tested:**
  - Sigmoid
  - Tanh
  - ReLU
  - Sigmoid–Tanh (mixed)
  - Tanh–ReLU (mixed)
  - Sigmoid–Tanh–ReLU (mixed)
- **Output Layer:** Softmax (10 digit classes)
- **Loss Function:** Negative Log Likelihood (computed from Softmax probabilities)
- **Optimizer:** Adam
- **Framework:** PyTorch

---

## 📊 Datasets Used
### MNIST Dataset
- Standard benchmark dataset for handwritten digit recognition.
- Automatically downloaded during execution.

### External / Custom Datasets
Used to evaluate real-world generalization:
- `handwritten`
- `online`
- `paint_whitebg`
- `word_blackbg`
- `word_whitebg`


---

## 🔬 Experiments Conducted
### Baseline Experiments
- Sigmoid activation (10 epochs)
- Tanh activation (10 epochs)

### Additional Experiments
- ReLU activation
- Mixed activation configurations:
  - Sigmoid → Tanh
  - Tanh → ReLU
  - Sigmoid → Tanh → ReLU
- Extended training up to **15 epochs**

---

## 🏆 Key Results (MNIST – 15 Epochs)
- **Sigmoid:** ~96.8%
- **Tanh:** ~97.6%
- **ReLU:** ~98.1% (best performance)
- **Mixed activations:** Competitive but not superior to pure ReLU

**Observation:**
- ReLU showed faster convergence and better gradient flow.
- Sigmoid and Tanh benefited only marginally from extended training.

---

## ⚠️ Generalization Findings
- High accuracy on MNIST does not fully translate to real-world data.
- Noticeable performance drops on handwritten and online digit datasets.
- Model sensitivity observed for:
  - Writing style variation
  - Image noise
  - Background differences

---

## 🧩 Ethical and Reliability Considerations
- Risk of misclassification in real-world applications.
- Potential bias toward writing styles present in training data.
- Accessibility challenges for users with non-standard handwriting.
- Importance of transparency and confidence-aware predictions.

---

## 🛠️ Mitigation Strategies
- Human-in-the-loop validation for low-confidence predictions.
- Confidence thresholding to reduce incorrect automated decisions.
- Expanding training data with diverse handwriting styles.
- Continuous monitoring after deployment.

---

## 🚀 Future Improvements
- Replace MLP with Convolutional Neural Networks (CNNs).
- Apply data augmentation techniques.
- Evaluate on larger and more diverse datasets.
- Introduce uncertainty estimation for predictions.

---

## ⚙️ Installation
**Requirements:**
- Python 3.8+
- pip

**Install dependencies:**
```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn pillow

