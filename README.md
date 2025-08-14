# Doubly-Stochastic-DGP
Deep Gaussian Processes with Doubly Stochastic Variational Inference 

This code accompanies the paper

@inproceedings{salimbeni2017doubly,
  title={Doubly stochastic variational inference for deep gaussian processes},
  author={Salimbeni, Hugh and Deisenroth, Marc},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}

See the arxiv version at https://arxiv.org/abs/1705.08933

This code now offers additional functionality than in the above paper. In particular, natural gradients are now supported. If you use these, please consider citing the following paper:

@inproceedings{salimbeni2018natural,
  title={Natural Gradients in Practice: Non-Conjugate Variational Inference in Gaussian Process Models},
  author={Salimbeni, Hugh and Eleftheriadis, Stefanos and Hensman, James},
  booktitle={Artificial Intelligence and Statistics},
  year={2018}
}

**Disclaimer:** This is a fork of the original codebase, updated to work with GPflow 2.x and TensorFlow 2 by [Boyuan Deng](https://bydeng01.github.io/).