
## OOD Detection and calibration with Temperature Scaling

For calibration with temperature scaling:

- **Score function:** `max_softmax`
- **Temperature range:** 0.5 to 10.0 (step 0.5)
- **Selection criterion:** lowest **ECE** (Expected Calibration Error), automatically selected.
- **In-distribution dataset:** CIFAR-10
- **Out-of-distribution dataset:** CIFAR-100

**Results:**

![ECE_Tscaling](../images/LAB4/ece_first_model.png "ECE calibration")
![Calibration](../images/LAB4/calibration_first_model.png "Calibration curves")
![hist before](../images/LAB4/hist_first_model.png "Hist before T scaling")
![hist after](../images/LAB4/hist_first_mode_after_scalingl.png "Hist after T scaling")

![ROC curve ID](../images/LAB4/ROC-ID-fistmodel.png "ROC curve")
![PR curve](../images/LAB4/PRfirst-model.png "PR OOD detection")

---

## FGSM Adversarial Attacks

For the FGSM task, a function was implemented to perform **untargeted** or **targeted** attacks, depending on the provided arguments (see [OOD_script](../deeo_learning_utils/src/OOD/OOD_utils.py)):

- Both **single sample** and **batch** versions are available.
- The **batch version** is used internally in the [Trainer_script](../deeo_learning_utils/src/Trainer/Trainer.py) class for adversarial training (private method).
- The **public method** is for single sample attacks.

**Examples:**

- Visualizations include:
  - Attacked image
  - Perturbation heatmap
  - Predicted label
  - Attack budget

**Attack budgets:**
- Untargeted: `0.01` to `0.09`
- Targeted: `0.01` to `0.19`

*Untargeted attacks*

![original image](../images/LAB4/original_imag.png "Original image")
![0.01 image](../images/LAB4/0.01_attack_untargeted.png "Image attacked with 0.01 budget")
![0.01 heatmap](../images/LAB4/0.01_heatmap_untargeted.png "Heatmap atatck with 0.01 budget")
![0.07 image](../images/LAB4/0.07_attack_untargeted.png "Image attacked with 0.07 budget")
![0.07 image](../images/LAB4/0.07_heatmap_untargeted.png "Heatmap atatck with 0.07 budget")

*Targeted attacks (target class: **bird**)*

![original image](../images/LAB4/original_image_targeted.png "Original image")
![0.01 image](../images/LAB4/0.01_targeted.png "Image attacked with 0.01 budget")
![0.01 heatmap](../images/LAB4/0.01_heatmpa_targeted.png "Heatmap atatck with 0.01 budget")
![0.07 image](../images/LAB4/0.05_targeted.png "Image attacked with 0.07 budget")
![0.07 image](../images/LAB4/0.05_hetamap_targeted.png "Heatmap atatck with 0.07 budget")


---

## Image Quality Degradation

To quantitatively assess image degradation for both attack types, the metric *PSNR* (Peak Signal-to-Noise Ratio) was used:

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

