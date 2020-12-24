#--------------------------------------
## LOAD PACKAGES
using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker
using YAML, Dates, BSON, Random, Statistics, Printf
using CUDA
using RegNeuralODE: accuracy
using Flux.Optimise: update!
using Flux: @functor, glorot_uniform, logitcrossentropy
using Tracker: TrackedReal, data
import Base.show

CUDA.allowscalar(false)
#--------------------------------------

#--------------------------------------
## CONFIGURATION
config_file = joinpath(pwd(), "experiments", "configs", "latent_ode.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

hparams = config["hyperparameters"]
BATCH_SIZE = hparams["batch_size"]
REGULARIZE = hparams["regularize"]
EPOCHS = hparams["epochs"]
EXPERIMENT_LOGDIR = joinpath(pwd(), "results", "latent_ode", "$(string(now()))_$REGULARIZE")
MODEL_WEIGHTS = joinpath(EXPERIMENT_LOGDIR, "weights.bson")
FILENAME = joinpath(EXPERIMENT_LOGDIR, "results.yml")

# Create a directory to store the results
isdir(EXPERIMENT_LOGDIR) || mkpath(EXPERIMENT_LOGDIR)
cp(config_file, joinpath(EXPERIMENT_LOGDIR, "config.yml"))
#--------------------------------------

#--------------------------------------
## NEURAL NETWORK
# Latent GRU Model
struct LatentGRU{U,R,N,D}
    update_gate::U
    reset_gate::R
    new_state::N
    latent_dim::D
end

function LatentGRU(in_dim::Int, h_dim::Int, latent_dim::Int)
    update_gate = Chain(
        Dense(latent_dim * 2 + in_dim * 2 + 1, h_dim, CUDA.tanh),
        Dense(h_dim, latent_dim, σ),
    )
    reset_gate = Chain(
        Dense(latent_dim * 2 + in_dim * 2 + 1, h_dim, CUDA.tanh),
        Dense(h_dim, latent_dim, σ),
    )
    new_state = Chain(
        Dense(latent_dim * 2 + in_dim * 2 + 1, h_dim, CUDA.tanh),
        Dense(h_dim, latent_dim * 2),
    )
    return LatentGRU(update_gate, reset_gate, new_state, latent_dim)
end

@functor LatentGRU

function single_run(p::LatentGRU, y_mean, y_std, x)
    # y -> latent_dim x B
    # x -> in_dim x B
    # x is the concatenation of a data and mask and Δtime
    y_concat = vcat(y_mean, y_std, x)

    update_gate = p.update_gate(y_concat)
    reset_gate = p.reset_gate(y_concat)

    concat = vcat(y_mean .* reset_gate, y_std .* reset_gate, x)

    new_state = p.new_state(concat)
    new_state_mean = new_state[1:p.latent_dim, :]
    # The easy-neural-ode paper uses abs, which is indeed a strange
    # choice. The standard is to predict log_std and take exp.
    new_state_std = abs.(new_state[p.latent_dim+1:end, :])

    new_y_mean = @. (1 - update_gate) * new_state_mean + update_gate * y_mean
    new_y_std = @. (1 - update_gate) * new_state_std + update_gate * y_std

    mask = sum(x[(size(x, 1)÷2+1):end, :], dims = 1) .> 0

    new_y_mean = @. mask * new_y_mean + (1 - mask) * y_mean
    new_y_std = @. abs(mask * new_y_std + (1 - mask) * y_std)

    return new_y_mean, new_y_std
end

function (p::LatentGRU)(x::AbstractArray{T,3}) where {T}
    z = TrackedArray(CUDA.zeros(T, p.latent_dim, size(x, 3)))
    y_mean, y_std = z, z
    for t = size(x, 2):-1:1
        y_mean, y_std = single_run(p, y_mean, y_std, x[:, t, :])
    end
    return vcat(y_mean, y_std)
end
#--------------------------------------

#--------------------------------------
## MODEL + DATASET + TRAINING UTILS
# Get the dataset
train_dataloader, test_dataloader =
    load_physionet(BATCH_SIZE, "data/physionet.bson", 0.9, x -> gpu(track(x)))

opt = Flux.Optimise.Optimiser(InvDecay(1.0e-5), Momentum(0.1, 0.9))

# Setup the models
gru_rnn = LatentGRU(37, 40, 50) |> track |> gpu
rec_to_gen = Chain(Dense(100, 50, CUDA.tanh), Dense(50, 2 * 20)) |> track |> gpu
gen_dynamics =
    Chain(
        x -> CUDA.tanh.(x),
        Dense(20, 50, CUDA.tanh),
        Dense(50, 20, CUDA.tanh),
        Dense(20, 50, CUDA.tanh),
        Dense(50, 20, CUDA.tanh),
        Dense(20, 50, CUDA.tanh),
        Dense(50, 20, CUDA.tanh),
        Dense(20, 50, CUDA.tanh),
        Dense(50, 20, CUDA.tanh),
    ) |>
    track |>
    gpu
node = TrackedNeuralODE(
    gen_dynamics,
    [0.0f0, 1.0f0],
    false,
    false,
    Tsit5(),
    saveat = train_dataloader.data[5][1, :, 1] |> cpu |> untrack,
    reltol = 1.4f-3,
    abstol = 1.4f-3,
)
gen_to_data = Dense(20, 37) |> track |> gpu

model = LatentTimeSeriesModel(gru_rnn, rec_to_gen, node, gen_to_data)
ps = Flux.trainable(model)

# Loss Functions
function log_likelihood(∇pred, mask)
    function _sum_stable_infer(x)::typeof(x)
        return sum(x, dims = (1, 2))
    end
    σ_ = 0.01f0
    sample_likelihood =
        -CUDA.pow.(∇pred, 2) ./ (2 * σ_^2) .- log(σ_) .- log(Float32(2π)) ./ 2
    return reshape(_sum_stable_infer(sample_likelihood) ./ _sum_stable_infer(mask), :)
end

# Holds only for a Standard Gaussian Prior
kl_divergence(μ, logσ²) =
    reshape(mean(exp.(logσ²) .+ CUDA.pow.(μ, 2) .- 1 .- logσ², dims = 1) ./ 2, :)

mse(∇pred) = mean(∇pred .^ 2)

function loss_function(data, mask, _t, model, p1, p2, p3, p4; λᵣ = 1.0f2, λₖ = 1.0f0)
    # TODO: Add regularization term
    x_ = vcat(data, mask, _t)
    result, μ₀, logσ², nfe, sv = model(x_, p1, p2, p3, p4)

    data_ = data .* mask
    pred_ = result .* mask
    ∇pred = pred_ .- data_

    _log_likelihood = log_likelihood(∇pred, mask)
    _kl_div = kl_divergence(μ₀, logσ²)

    return mean(_log_likelihood .- λₖ .* _kl_div)
end
#--------------------------------------

#--------------------------------------
## TESTING THE MODEL
# Dummy Input for first run
d, m, _, _, t, _ = iterate(train_dataloader)[1]
_t = hcat(t[:, 2:end, :] .- t[:, 1:end-1, :], TrackedArray(CUDA.zeros(1, 1, size(t, 3))))
x_ = vcat(d, m, _t)
result, μ₀, logσ², nfe, sv = model(x_)
loss_function(d, m, _t, model, ps...)

Tracker.gradient((p1, p2, p3, p4) -> loss_function(d, m, _t, model, p1, p2, p3, p4), ps...)
#--------------------------------------
