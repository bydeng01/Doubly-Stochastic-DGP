# Deep Gaussian Processes with Doubly Stochastic Variational Inference

This repository is a fork of the original codebase, updated by Boyuan Deng to be compatible with GPflow 2.x and TensorFlow 2. The updated version will be used to provide support for [Emukit](https://github.com/bydeng01/emukit).

If you use this code, please cite the original paper listed in the **Citation** section.
To support my work, consider giving the repository a ⭐ on GitHub.

## Installation

```bash
python -m pip install -U git+https://github.com/bydeng01/Doubly-Stochastic-DGP.git
```

This version targets Python 3.9-3.12 with GPflow 2.10, TensorFlow 2.16, and TensorFlow Probability 0.24.
For dataset download helpers, install the optional dataset extra:

```bash
python -m pip install -U "git+https://github.com/bydeng01/Doubly-Stochastic-DGP.git#egg=Doubly-Stochastic-DGP[datasets]"
```

For demo notebooks and scripts, install the demo extra:

```bash
python -m pip install -U "git+https://github.com/bydeng01/Doubly-Stochastic-DGP.git#egg=Doubly-Stochastic-DGP[demos]"
```

## Citation
This code accompanies the paper https://arxiv.org/abs/1705.08933
```bibtex
@inproceedings{salimbeni2017doubly,
  title={Doubly stochastic variational inference for deep gaussian processes},
  author={Salimbeni, Hugh and Deisenroth, Marc},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

This code now offers additional functionality than in the above paper. In particular, natural gradients are now supported. If you use these, please consider citing the following paper:

```bibtex
@inproceedings{salimbeni2018natural,
  title={Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models},
  author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
  booktitle={Artificial Intelligence and Statistics},
  year={2018}
}
```
