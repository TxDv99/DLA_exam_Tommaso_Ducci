# Laboratory 1 — Custom Trainer, MyResNet API, and Experiments on CIFAR-10/100

## Overview

In this first laboratory, I focused on the construction of a comprehensive `Trainer` class and its methods to facilitate downstream tasks throughout all other laboratories.  
The class implements a variety of functionalities (see the README in the [main folder](../README.md)).

Alongside the `Trainer` class, I implemented a class **`MyResNet`**, designed as an easy-to-use API for building neural networks composed of:

- Convolutional layers  
- Batch normalization layers  
- Dropout layers  
- Linear layers  
- Activation functions  

---

## MyResNet — Design and Features

The API takes arguments in the form of tuples:

```
(layer_type, param1, param2, ...)
```
Example:  
```python
("Linear", in_size, hidden_size, out_size)
```

It also supports **skip connections** in dictionary format:
```python
{index_from: index_to}
```

### Public Methods
- `addConv`
- `addLinear`
- `get_submodel`
- `test` — includes optional confusion matrix plotting (`plot=True`).

### Shape Mismatch Handling
If there is a mismatch between:
- Sequential layers, or
- Layers connected via skip connections,  

the class automatically **reprojects** using flattening followed by a linear layer.

**Note:** To create skip connections and handle reprojections, the **data shape must be provided at initialization**.

---

## Task 1 — MLP Experiment

**Architecture:**  
- Depth: 3  
- Hidden size: 64  

**Training parameters:**  
- Epochs: up to 25  
- Early stopping: patience = 2 (check validation every epoch)  
- Batch size: 256  
- Learning rate: 0.001  
- Optimizer: Adam  
- Weight decay: 1e-4  

**Motivation:**  
Observed that deeper nets without skip connections tend to overfit.

**Results:**  
<!-- Inserire qui tabella o grafico con performance su train/validation/test -->

---

## Task 2 — MLP Depth and Skip Connections

Repeated the experiment for depths **3, 5, 7, 9**:

- With skip connections  
- Without skip connections  
- Stricter early stopping policy  

**Observation:**  
Networks with skip connections **consistently outperformed** their counterparts.

**Results:**  
![Train MLP](DLA_exam_Tommaso_Ducci/images/LAB1/various_depth_trainMLP.png "Losses and accs MLP various depths")
![Val MLP](DLA_exam_Tommaso_Ducci/images/LAB1/various_depth_valMLP.png "Losses and accs MLP various depths")

---

## Task 3 — ResNet Architectures

**Architectures tested:**
1. Small ResNet without skip connections (**benchmark**)
2. Medium ResNet with skip connections
3. Medium ResNet without skip connections
4. Deeper ResNet without skip connections
5. Deeper ResNet with skip connections

**Results:**
- Networks with skip connections prevailed in:
  - Performance
  - Number of epochs before early stopping  
- Best model: Deeper ResNet with skip connections → **Accuracy: 0.885 on CIFAR-10 test set**

**Training parameters:**
- Optimizer: Adam (`lr = 0.0001`)
- Batch size: 64
- Epochs: up to 65
- Early stopping: 3 validations without improvement (validation checked every 3 epochs)
- Train/validation split: 80%/20%
- Data augmentation: applied to 40% of the training set

**Data Augmentation:**
<!-- Inserire qui tabella con le trasformazioni di data augmentation -->

---

## Task 4 — Fine-Tuning on CIFAR-100

1. **Baseline:**  
   - Features extracted from the **last convolutional layer** of the best CNN (from previous task)
   - SVM with **RBF kernel**  
   - Observed accuracy: *[Inserire valore]*  
   <!-- Inserire qui eventuale immagine/grafico della baseline -->

2. **Fine-tuning Protocol:**  
   - Progressive unfreezing of layers  
   - Different optimizers applied at different stages  
   - Goal: maximize transfer performance from CIFAR-10 to CIFAR-100

**Results:**  
<!-- Inserire qui grafici e tabelle prima/dopo fine-tuning -->

---


