# RegNeuralDE

Official Implementation of the *ICML 2021* Paper [**Opening the Blackbox: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics**](http://proceedings.mlr.press/v139/pal21a.html)

## USAGE

Experiments provided here were developed and tested on Julia v1.5.3. All other package versions are automatically enforced. To install do the following in Julia REPL:

```julia
] dev https://github.com/avik-pal/RegNeuralDE.jl
```

The code will be downloaded in the `JULIA_PKG_DEVDIR` directory.


## CITATION

If you found this codebase useful in your research, please consider citing

```bibtex
@InProceedings{pmlr-v139-pal21a,
  title = 	 {Opening the Blackbox: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics},
  author =       {Pal, Avik and Ma, Yingbo and Shah, Viral and Rackauckas, Christopher V},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {8325--8335},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/pal21a/pal21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/pal21a.html},
}

```

## DATASETS

* Preprocessed Physionet Data can be downloaded from [here](https://github.com/avik-pal/RegNeuralDE.jl/releases/download/v0.1.0/physionet.zip). Place the downloaded file in `data/physionet.bson`.

## EXPERIMENTS

Important Parameters of the Experiments are controlled using the `yml` files in `experiments/configs`.

### SUPERVISED CLASSIFICATION USING NEURAL ODE

Parameters controlled by `experiments/configs/mnist_node.yml`. To train a Vanilla/Regularized Neural ODE for MNIST classification:

```bash
$ julia --project=. experiments/mnist_node.jl
```

### LATENT ODE FOR TIME SERIES INTERPOLATION

Parameters controlled by `experiments/configs/latent_ode.yml`. To train a Vanilla/Regularized Latent ODE with GRU Encoder for Physionet Time Series Interpolation

```bash
$ julia --project=. experiments/latent_ode.jl
```

### TOY NEURAL SDE

To train a Vanilla and Regularized Neural SDE

```bash
$ julia --project=. experiments/sde_toy_problem.jl
```

### SUPERVISED CLASSIFICATION USING NEURAL SDE

Parameters controlled by `experiments/configs/mnist_nsde.yml`. To train a Vanilla/Regularized Neural ODE for MNIST classification:

```bash
$ julia --project=. experiments/mnist_nsde.jl
```

