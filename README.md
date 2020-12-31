# RegNeuralODE

Regularizing Neural ODEs to make them easier to solve during evaluation

## USAGE

Experiments provided here were developed and tested on Julia v1.5.3. All other package versions are automatically enforced. To install do the following in Julia REPL:

```julia
] dev https://github.com/avik-pal/RegNeuralODE.jl
```

The code will be downloaded in the `JULIA_PKG_DEVDIR` directory.

## DATASETS

* **MINIBOONE**: Download the preprocessed data from https://github.com/gpapamak/maf. We only need the MINIBOONE Dataset here. Place it in `data/miniboone.npy`.
* **PHYSIONET**: Download the `physionet.bson` file from the initial release of the project and place it in `data/physionet.bson`.

## EXPERIMENTS

Important Parameters of the Experiments are controlled using the `yml` files in `experiments/configs`.

### SUPERVISED CLASSIFICATION

Parameters controlled by `experiments/configs/mnist_node.yml`. To train a Vanilla/Regularized Neural ODE for MNIST classification:

```bash
$ julia experiments/mnist_node.jl
```

### LATENT ODE FOR TIME SERIES INTERPOLATION

Parameters controlled by `experiments/configs/latent_ode.yml`. To train a Vanilla/Regularized Latent ODE with GRU Encoder for Physionet Time Series Interpolation

```bash
$ julia experiments/latent_ode.jl
```