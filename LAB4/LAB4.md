
## OOD Detection with Temperature Scaling

For calibration with temperature scaling:

- **Score function:** `max_softmax`
- **Temperature range:** 0.5 to 10.0 (step 0.5)
- **Selection criterion:** lowest **ECE** (Expected Calibration Error), automatically selected.
- **In-distribution dataset:** CIFAR-10
- **Out-of-distribution dataset:** CIFAR-100

**Results:**
<!-- Insert plots here: ECE before/after, confusion matrices -->
- Confusion matrix and performance metrics are shown **before** and **after** temperature scaling.

---

## FGSM Adversarial Attacks

For the FGSM task, a function was implemented to perform **untargeted** or **targeted** attacks, depending on the provided arguments (see `deep_learning_utils`):

- Both **single sample** and **batch** versions are available.
- The **batch version** is used internally in the `Trainer` class for adversarial training (private method).
- The **public method** is for single sample attacks.

**Examples:**
<!-- Insert example images and perturbation heatmaps -->
- Visualizations include:
  - Attacked image
  - Perturbation heatmap
  - Predicted label
  - Attack budget

**Attack budgets:**
- Untargeted: `0.001` to `0.05`
- Targeted: `0.002` to `0.022`

---

## Image Quality Degradation

To quantitatively assess image degradation for both attack types, the following metrics were used:

- **L∞ norm**
- **PSNR** (Peak Signal-to-Noise Ratio)

---

## Adversarial Training

Training with **untargeted adversarial attacks** was performed with a budget of `0.05`.

- Reported results include:
  - Confusion matrix of the model trained **without** adversarial training
  - Confusion matrix of the model trained **with** adversarial training

<!-- Insert comparison confusion matrices here -->

---

## Conclusions

Temperature scaling significantly improved calibration in the OOD task, reducing ECE without degrading accuracy.  
FGSM attacks demonstrated the vulnerability of the model to both targeted and untargeted perturbations, even at small budgets.  
Adversarial training with FGSM at ε = 0.05 increased robustness against untargeted attacks, but caused a slight drop in clean accuracy — a known trade-off in robust training.

