# MicroPlant: Efficient Plant Disease Classification

A lightweight deep learning pipeline for plant disease classification, optimized for **edge deployment** through **knowledge distillation, pruning, and quantization**.

---

## Project Overview

This project builds an end-to-end machine learning pipeline:

* **Exploratory Data Analysis (EDA)**
* **Model Training with Knowledge Distillation**
* **Model Compression (Pruning + Quantization)**

The goal is to create a model that is both:

* **Accurate**
* **Efficient (small & fast)**

---

## Key Techniques

### Knowledge Distillation

A smaller model (MicroPlant) learns from a larger teacher model (ResNet18), improving performance without increasing complexity.

### Pruning

Removes less important weights to reduce model size.

### Quantization (QAT)

Reduces precision (e.g., FP32 → INT8) to improve inference speed and efficiency.

---

## Project Structure

```
MicroPlant/
│
├── data/                  # Dataset (not included or partial)
├── models/                # Saved model weights
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Model_Compression.ipynb
│
├── src/
│   ├── architectures.py   # Model definitions
│   ├── preprocessing.py   # Data loading & transforms
│   ├── training.py        # Training + distillation
│   ├── compression.py     # Pruning + quantization
│   └── utils.py           # Utilities
│
└── README.md
```

---

## Notebooks

### 01 — Data Exploration

* Dataset distribution
* Class balance
* Sample visualization

---

### 02 — Model Training

* Baseline MicroPlant model
* Teacher model (ResNet18)
* Knowledge Distillation
* Performance comparison

---

### 03 — Model Compression

* Global pruning
* Fine-tuning after pruning
* Quantization-Aware Training (QAT)
* Efficiency vs performance trade-off

---

## 📈 Results (Example)

| Model          | F1 Score | Size | Notes             |
| -------------- | -------- | ---- | ----------------- |
| Baseline       | -        | -    | Lightweight model |
| + Distillation | ↑        | -    | Better accuracy   |
| + Pruning      | ~        | ↓    | Smaller model     |
| + Quantization | ~        | ↓↓   | Deployment-ready  |

---

## Installation

```bash
git clone https://github.com/axolotl-01/MicroPlant.git
cd MicroPlant
pip install -r requirements.txt
```

---

## Usage

Run notebooks in order:

```bash
notebooks/01_Data_Exploration.ipynb
notebooks/02_Model_Training.ipynb
notebooks/03_Model_Compression.ipynb
```

>  If notebooks fail to render on GitHub, open them locally in VS Code with the Jupyter extension.

---

## Highlights

* Custom lightweight CNN (MicroPlant)
* Knowledge distillation pipeline
* Structured pruning implementation
* Quantization-aware training (QAT)
* Designed for edge deployment

---

## Future Improvements

* Add real-time inference demo
* Deploy to mobile or edge device
* Hyperparameter optimization
* Larger dataset support

---

## License

This project is for educational and research purposes.

---

## Acknowledgements

* PyTorch
* Torchvision
* Plant disease datasets (public sources)
