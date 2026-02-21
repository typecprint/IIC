# Invariant Information Clustering (IIC)

This project implements the **Invariant Information Clustering (IIC)** algorithm using PyTorch. IIC is an unsupervised learning method that clusters data by maximizing mutual information between original images and their augmented (transformed) versions.

## Research & Theory

### Paper Reference

This repository is based on the research presented in:

> **Invariant Information Clustering for Unsupervised Image Classification and Segmentation**
> Xu Ji, João F. Henriques, Andrea Vedaldi
> _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019_
> [arXiv:1807.06653](https://arxiv.org/abs/1807.06653)

### Theoretical Background

IIC aims to learn a function $\Phi$, represented by a deep neural network, that maps an input image $x$ and its augmented counterpart $x'$ into a probability distribution over $C$ clusters.
The core objective is to **maximize the Mutual Information (MI)** between the cluster assignment predictions of $x$ and $x'$.

Let $\Phi(x)$ and $\Phi(x')$ denote the output probability mass functions over the target classes. By treating the predicted cluster assignments as random variables $Z$ and $Z'$, the training objective is to maximize their mutual information $I(Z; Z')$:

$$ I(Z; Z') = H(Z) - H(Z | Z') = \sum*{c=1}^C \sum*{c'=1}^C P(z=c, z'=c') \ln \frac{P(z=c, z'=c')}{P(z=c) P(z'=c')} $$

Where:

- $\mathbf{P}$ is the $C \times C$ joint probability matrix, computed by averaging $\Phi(x) [\Phi(x')]^T$ over a batch.
- To enforce symmetry (as the order of $x$ and $x'$ is interchangeable), the joint distribution is symmetrized: $\mathbf{P}\_{sym} = \frac{\mathbf{P} + \mathbf{P}^T}{2}$.
- The marginal distributions $P(z=c)$ and $P(z'=c')$ are obtained by summing the joint probability matrix over its rows and columns, respectively.

By maximizing mutual information, the network is naturally encouraged to:

1. **Maximize Predictability:** Ensure that predictions for different augmented views of the same image are consistent, thereby reducing the conditional entropy $H(Z | Z')$.
2. **Promote Uniformity:** Distribute cluster assignments uniformly across all $C$ clusters, avoiding trivial solutions where all images are assigned to a single, identical class (which maximizes the marginal entropy $H(Z)$).

## Overview

The repository consists of the following structure:

- `main.py` - The entry point for the training and testing loop. It handles argument parsing, dataset initialization, logging, and evaluation.
- `model.py` - Contains the `ModelIIC` definition (using a modified MobileNetV2 backbone) and the `IID_loss` calculation function.
- `dataset.py` - Manages data downloading and loading using `torchvision.datasets.STL10` and applies necessary data augmentations (cropping, color jittering, rotation, flipping) to generate the invariant pairs.
- `requirements.txt` - Lists the necessary Python packages.

## Requirements

Before running the code, make sure to install the required dependencies inside a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The implementation uses the **STL-10** dataset by default.
The `dataset.py` script automatically downloads the dataset (approx. 2GB) to the `./data` directory upon running `main.py` for the first time. You do not need to extract any binary files manually.

## Usage

To train the model and evaluate the clustering results, simply execute:

```bash
python main.py
```

### Notes on Training Strategy

- The code automatically detects if a GPU (`cuda`) is available and runs on it to accelerate the training process. **Running on a multi-GPU environment or powerful single GPU is highly recommended for full dataset training.**
- The `epochs` parameter is set to `3` by default within `main.py`. Change this value inside the script if you plan to train for longer iterations to get optimal clustering performance.
- During training, the progression, batch losses, and execution times are securely logged out to the console and to a local file named `iic_training.log`.
- After training, the script outputs an aggregated confusion matrix connecting actual labels with clustered assignments and saves the learned weights to `./model/resnet/model.pt`.
