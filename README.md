# Communication Efficient Private Federated Learning Using Dithering

This repository contains the source code for the experiments of the paper "Communication Efficient Private Federated Learning Using Dithering", which is recently accepted to ICASSP 2024. The preprint is available [here] (https://arxiv.org/pdf/2309.07809.pdf)

### Installation
To install, run `pip install -r requirements.txt`

### Fixing the Opacus

There is currently a bug in Opacus library. It gives an error when the data loader samples an empty batch using a GPU. To fix it please modify the clip_and_accumulate function in opacus/optimizers/optimizer.py such that the line 399 changed from

per_sample_clip_factor = torch.zeros((0,))

to

per_sample_clip_factor = torch.zeros((0,), device=self.grad_samples[0].device)

or simply replace opacus/optimizers/optimizer.py with optimizer.py file that we provide.

### Running
To run the experiments, please set the parameters properly on options.py, and then run `python3 src/federated_main.py`

To calculate the privacy parameters, i.e., epsilon and delta, run `python3 src/privacy_accounting.py`
