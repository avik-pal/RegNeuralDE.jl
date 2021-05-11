# RegNeuralDE

Official Implementation of the *ICML 2021* Paper [**Opening the Blackbox: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics**](https://arxiv.org/abs/2105.03918)

## USAGE

Experiments provided here were developed and tested on Julia v1.5.3. All other package versions are automatically enforced. To install do the following in Julia REPL:

```julia
] dev https://github.com/avik-pal/RegNeuralDE.jl
```

The code will be downloaded in the `JULIA_PKG_DEVDIR` directory.


## CITATION

If you found this codebase useful in your research, please consider citing

```bibtex
@inproceedings{
    pal2021opening,
    title={{O}pening the {B}lackbox: {A}ccelerating {N}eural {D}ifferential {E}quations by {R}egularizing {I}nternal {S}olver {H}euristics},
    author={Avik Pal and Yingbo Ma and Viral B. Shah and Christopher Rackauckas},
    booktitle={International Conference on Machine Learning},
    year={2021},
    eprint={2105.03918},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
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

