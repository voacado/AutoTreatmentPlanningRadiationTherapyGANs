# Automated Treatment Planning Using Generative Adversarial Networks

## Inspiration

Model Architecture:
Mahmood, R., Babier, A., McNiven, A., Diamant, A. &amp; Chan, T.C.Y.. (2018). Automated Treatment Planning in Radiation Therapy using Generative Adversarial Networks. <i>Proceedings of the 3rd Machine Learning for Healthcare Conference</i>, in <i>Proceedings of Machine Learning Research</i> 85:484-499 Available from https://proceedings.mlr.press/v85/mahmood18a.html.

Data Source (CORT):
Craft, D., Bangert, M., Long, T., Papp, D., & Unkelbach, J. (2014). Shared data for intensity modulated radiation therapy (IMRT) optimization research: the CORT dataset. Gigascience, 3(1). https://doi.org/10.1186/2047-217x-3-37

## Introduction

This model uses Generative Adversarial Networks to properly determine where to apply radiation therapy (as PTV, or planning target volume). The model utilizes a pix2pix architecture to convert the CT slice into a PTV suggestion.

## Requirements

To run this notebook, make sure you have the following packages installed:

```
numpy
scipy
pydicom
matplotlib
skimage
torch
```

Specifically, for the following imports:

```
import numpy as np
import scipy.io
import pydicom 

import matplotlib.pyplot as plt
from skimage.transform import resize as sk_resize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

Additionally, make sure your dataset is extracted in your `dataset` folder such that the file directory is as such:
- `dataset/LIVER/`
- `dataset/Liver_dicom/`

NOTE: I ran this model completely on Kaggle. All the required packages are already pre-installed on their compute, and uploading the dataset is very straightforward. Make sure you change the variable `kaggle_path` so it points to your dataset!
