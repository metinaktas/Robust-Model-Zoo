# Robust-Model-Zoo
This repository includes various adversarially robust model checkpoints and their robustness metrics.

## CIFAR-10
The robustness metrics for [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) are measured by using [AutoAttack](https://github.com/fra31/auto-attack) repository.


The links for the checkpoints for different models are listed below.


| Model File Name  | Checkpoint | Complexity |
| ---------------- | ------------- | ------------- |
| [model_1](https://github.com/metinaktas/Robust-Model-Zoo/blob/main/CIFAR-10/Models/tf1/model_1/model.py)  | [download](https://drive.google.com/uc?export=download&id=16PLzHqErmNdPHexlPfZ8ccxeIaqWOMTt) | 176 MB |
| [model_2](https://github.com/metinaktas/Robust-Model-Zoo/blob/main/CIFAR-10/Models/tf1/model_2/model.py) | [download](https://drive.google.com/uc?export=download&id=1s76yDCBK86KN_cxjJM87i4qn897arAY_) | 104 MB |

### Results

The robust accuracy values for various models are listed below.

The proposed models in this repository contain "quantization" blocks for realizing gradient masking. To evaluate the robustness of the proposed models fairly, we should take into account the existence of "quantization" blocks and help backpropagation to find gradients. The [AutoAttack flag_docs](https://github.com/fra31/auto-attack/blob/master/flags_doc.md) suggest to use [BPDA](http://proceedings.mlr.press/v80/athalye18a.html) method, which approximates such functions with differentiable counterparts. Since BPDA is not implemented in AutoAttack, we modify the attack codes to handle different models in feedforward and backpropagation phases. With these modifications, we can use original proposed model (with quantization blocks) in evalution and model without quantization blocks in computing gradients during adversarial example generation. We reported robust accuracies for both with and without BPDA method.

| Model File Name  | Clean Accuracy | Robust Accuracy without BPDA| Robust Accuracy with BPDA|
| ---------------- | ------------- | ------------- | ------------- |
| [model_1](https://github.com/metinaktas/Robust-Model-Zoo/blob/main/CIFAR-10/Models/tf1/model_1/model.py)  | 77.10  | 62.20  | 53.30  |
| [model_2](https://github.com/metinaktas/Robust-Model-Zoo/blob/main/CIFAR-10/Models/tf1/model_2/model.py)  | 69.30  | 49.70  | 43.10  |
