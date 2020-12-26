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
    # The easy-neural-ode paper uses abs, which is indeed a strange choice.
    # Instead we treat this value as logσ²
    new_state_std = new_state[p.latent_dim+1:end, :]

    new_y_mean = @. (1 - update_gate) * new_state_mean + update_gate * y_mean
    new_y_std = @. (1 - update_gate) * new_state_std + update_gate * y_std

    mask = sum(x[(size(x, 1)÷2+1):end, :], dims = 1) .> 0

    new_y_mean = @. mask * new_y_mean + (1 - mask) * y_mean
    new_y_std = @. mask * new_y_std + (1 - mask) * y_std

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
    load_physionet(BATCH_SIZE, "data/physionet.bson", 0.8, x -> gpu(track(x)))

opt = Flux.Optimise.Optimiser(InvDecay(1e-5), AdaMax(0.01))

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
    REGULARIZE,
    Tsit5(),
    saveat = train_dataloader.data[5][1, :, 1] |> cpu |> untrack,
    reltol = 1.4f-8,
    abstol = 1.4f-8,
)
gen_to_data = Dense(20, 37) |> track |> gpu

model = LatentTimeSeriesModel(gru_rnn, rec_to_gen, node, gen_to_data)
ps = Flux.trainable(model)

# Anneal the regularization so that it doesn't overpower the
# the main objective
λᵣ₀ = 1.0f3
λᵣ₁ = 1.0f2
kᵣ = log(λᵣ₀ / λᵣ₁) / EPOCHS
# Exponential Decay
λᵣ_func(t) = λᵣ₀ * exp(-kᵣ * t)

λₖ_func(t) = max(0, 1 - 0.99f0^(t - 10))

## Loss Functions
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

function loss_function(
    data,
    mask,
    _t,
    model,
    p1,
    p2,
    p3,
    p4;
    λᵣ = 1.0f2,
    λₖ = 1.0f0,
    notrack = false,
)
    x_ = vcat(data, mask, _t)
    result, μ₀, logσ², nfe, sv = model(x_, p1, p2, p3, p4)

    data_ = data .* mask
    pred_ = result .* mask
    ∇pred = pred_ .- data_

    _log_likelihood = log_likelihood(∇pred, mask)
    _kl_div = λₖ .* kl_divergence(μ₀, logσ²)
    reg = REGULARIZE ? λᵣ * mean(sv.saveval) : zero(eltype(pred_))
    total_loss = -mean(_log_likelihood .- _kl_div) + reg

    if !notrack
        ll_un = -mean(_log_likelihood |> untrack)
        kl_un = mean(_kl_div |> untrack)
        rg_un = reg |> untrack
        tl_un = total_loss |> untrack
        logger(
            false,
            Dict(
                "Total Loss" => tl_un,
                "Negative Log Likelihood" => ll_un,
                "KL Divergence" => kl_un,
                "Our Regularization" => rg_un,
            ),
        )
    end

    return total_loss
end

function total_loss_on_dataset(model, dataloader)
    loss = 0.0f0
    count = 0
    for (i, (d, m, _, _, _, _)) in enumerate(dataloader)
        x_ = vcat(d, m, _t)
        result, _, _, _, _ = model(x_)

        data_ = d .* m
        pred_ = result .* m
        ∇pred = pred_ .- data_

        count += size(d, 3)
        loss += sum(sum(∇pred .^ 2, dims = (1, 2)) ./ sum(m, dims = (1, 2))) |> untrack
    end
    return loss ./ count
end
#--------------------------------------

#--------------------------------------
## LOGGING UTILITIES
nfe_counts = Vector{Float64}(undef, EPOCHS + 1)
train_loss = Vector{Float64}(undef, EPOCHS + 1)
test_loss = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes = Vector{Float64}(undef, EPOCHS + 1)  # The first value is a dummy value
inference_runtimes = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes[1] = 0

logger = table_logger(
    [
        "Epoch Number",
        "NFE Count",
        "Train Loss",
        "Test Loss",
        "Train Runtime",
        "Inference Runtime",
    ],
    ["Total Loss", "Negative Log Likelihood", "KL Divergence", "Our Regularization"],
)
#--------------------------------------

#--------------------------------------
## TESTING THE MODEL
# Dummy Input for first run
d, m, _, _, t, _ = iterate(train_dataloader)[1]
_t =
    hcat(t[:, 2:end, :] .- t[:, 1:end-1, :], TrackedArray(CUDA.zeros(1, 1, size(t, 3)))) |>
    untrack
x_ = vcat(d, m, _t |> track)
dummy_data = x_
result, μ₀, logσ², _, sv = model(x_)

loss_function(d, m, _t |> track, model, ps...; notrack = true)

Tracker.gradient(
    (p1, p2, p3, p4) ->
        loss_function(d, m, _t |> track, model, p1, p2, p3, p4; notrack = true),
    ps...,
)
#--------------------------------------

#--------------------------------------
## TRAINING
for epoch = 1:EPOCHS
    λᵣ = λᵣ_func(epoch - 1)
    λₖ = λₖ_func(epoch - 1)

    start_time = time()

    for (i, (d, m, _, _, _, _)) in enumerate(train_dataloader)
        gs = Tracker.gradient(
            (p1, p2, p3, p4) -> loss_function(
                d,
                m,
                _t |> track,
                model,
                p1,
                p2,
                p3,
                p4;
                λᵣ = λᵣ,
                λₖ = λₖ,
                notrack = false,
            ),
            ps...,
        )
        for (p, g) in zip(ps, gs)
            length(p) == 0 && continue
            update!(opt, data(p), data(g))
        end
    end

    # Record the time per epoch
    train_runtimes[epoch+1] = time() - start_time

    # Record the NFE count
    start_time = time()
    _, _, _, nfe, _ = model(dummy_data)
    inference_runtimes[epoch+1] = time() - start_time
    nfe_counts[epoch+1] = nfe

    # Test and Train Accuracy
    train_loss[epoch+1] = total_loss_on_dataset(model, train_dataloader)
    test_loss[epoch+1] = total_loss_on_dataset(model, test_dataloader)

    logger(
        false,
        Dict(),
        epoch,
        nfe_counts[epoch+1],
        train_loss[epoch+1],
        test_loss[epoch+1],
        train_runtimes[epoch+1],
        inference_runtimes[epoch+1],
    )
end
logger(true, Dict())
#--------------------------------------

#--------------------------------------
## STORE THE RESULTS
results = Dict(
    :nfe_counts => nfe_counts,
    :train_loss => train_loss,
    :test_loss => test_loss,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes,
)

weights = Flux.params(model) .|> cpu .|> untrack
BSON.@save MODEL_WEIGHTS weights

YAML.write_file(FILENAME, results)
#--------------------------------------
