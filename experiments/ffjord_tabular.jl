using RegNeuralODE, OrdinaryDiffEq, Flux, DiffEqFlux, Tracker, Random, Statistics
using ProgressLogging, YAML, Dates, BSON
using CUDA
using RegNeuralODE: loglikelihood
using Flux.Optimise: update!
using Flux: @functor, glorot_uniform, logitcrossentropy
using Tracker: TrackedReal, data
import Base.show

## Training Parameters
config_file = joinpath(pwd(), "experiments", "configs", "ffjord_tabular.yml")
config = YAML.load_file(config_file)

Random.seed!(config["seed"])

DATA_PATH = config["data_path"]
hparams = config["hyperparameters"]
BATCH_SIZE = hparams["batch_size"]
REGULARIZE = hparams["regularize"]
HIDDEN_DIMS = hparams["hidden_dims"]
INPUT_DIMS = hparams["input_dims"]
LR = hparams["lr"]
EPOCHS = hparams["epochs"]
EXPERIMENT_LOGDIR = joinpath(config["log_dir"], "$(string(now()))_$REGULARIZE")
MODEL_WEIGHTS = joinpath(EXPERIMENT_LOGDIR, "weights.bson")
FILENAME = joinpath(EXPERIMENT_LOGDIR, "results.yml")

# Create a directory to store the results
isdir(EXPERIMENT_LOGDIR) || mkpath(EXPERIMENT_LOGDIR)
cp(config_file, joinpath(EXPERIMENT_LOGDIR, "config.yml"))


struct ConcatSquashLayer{L, B, G}
    linear::L
    hyper_bias::B
    hyper_gate::G
end

function ConcatSquashLayer(in_dim::Int, out_dim::Int)
    l = Dense(in_dim, out_dim)
    b = Linear(1, out_dim)
    g = Dense(1, out_dim)
    return ConcatSquashLayer(l, b, g)
end

@functor ConcatSquashLayer

function (csl::ConcatSquashLayer)(x, t)
    _t = reshape(Tracker.collect([t]), 1, 1)
    return csl.linear(x) .* σ.(csl.hyper_gate(_t)) .+ csl.hyper_bias(_t)
end


struct MLPDynamics{L1, L2, L3}
    l1::L1
    l2::L2
    l3::L3
end

MLPDynamics(in_dims::Int, hdim1::Int, hdim2::Int) =
    MLPDynamics(ConcatSquashLayer(in_dims, hdim1),
                ConcatSquashLayer(hdim1, hdim2),
                ConcatSquashLayer(hdim2, in_dims))

@functor MLPDynamics

function (mlp::MLPDynamics)(x, t)
    x = softplus.(mlp.l1(x, t))
    x = softplus.(mlp.l2(x, t))
    return mlp.l3(x, t)
end


train_dataloader, test_dataloader = load_miniboone(BATCH_SIZE, DATA_PATH)


nn_dynamics = MLPDynamics(INPUT_DIMS, HIDDEN_DIMS, HIDDEN_DIMS)
ffjord = TrackedFFJORD(nn_dynamics |> track, [0.0f0, 1.0f0], REGULARIZE,
                       INPUT_DIMS, Tsit5(), save_everystep = false,
                       reltol = 6f-5, abstol = 6f-5,
                       save_start = false) |> track

if REGULARIZE
    function loss_function(x, model, p; λ = 1.0f2)
        pred, sv, sol = model(x, p)
        return -mean(pred) + λ * sum(sv.saveval)
    end
else
    function loss_function(x, model, p)
        pred, _, _, sol = model(x, p)
        return -mean(pred)
    end
end

nfe_counts = Vector{Float64}(undef, EPOCHS + 1)
train_loglikelihood = Vector{Float64}(undef, EPOCHS + 1)
test_loglikelihood = Vector{Float64}(undef, EPOCHS + 1)
train_runtimes = Vector{Float64}(undef, EPOCHS + 1)  # The first value is a dummy value
inference_runtimes = Vector{Float64}(undef, EPOCHS + 1)

train_runtimes[1] = 0

ps = ffjord.p

dummy_data = rand(Float32, 43, 128) |> track
start_time = time()
sol = ffjord(dummy_data)[end]
inference_runtimes[1] = time() - start_time
train_runtimes[1] = 0.0
nfe_counts[1] = sol.destats.nf
train_loglikelihood[1] = loglikelihood(ffjord, train_dataloader)
test_loglikelihood[1] = loglikelihood(ffjord, test_dataloader)
@info (train_runtimes[1], inference_runtimes[1], nfe_counts[1], train_loglikelihood[1], test_loglikelihood[1])

@progress for epoch in 1:EPOCHS
    start_time = time()

    @progress for (i, x) in enumerate(train_dataloader)
        gs = Tracker.gradient(p -> loss_function(x, ffjord, p), ps)[1]
        update!(opt, data(ps), data(gs))
    end
    # Record the time per epoch
    train_runtimes[epoch + 1] = time() - start_time

    # Record the NFE count
    start_time = time()
    sol = ffjord(dummy_data)[end]
    inference_runtimes[epoch + 1] = time() - start_time
    nfe_counts[epoch + 1] = sol.destats.nf

    train_loglikelihood[1] = loglikelihood(ffjord, train_dataloader)
    test_loglikelihood[1] = loglikelihood(ffjord, test_dataloader)
    @info (train_runtimes[epoch + 1], inference_runtimes[epoch + 1], nfe_counts[epoch + 1], train_loglikelihood[epoch + 1], test_loglikelihood[epoch + 1])
end

results = Dict(
    :nfe_counts => nfe_counts,
    :train_likelihood => train_likelihood,
    :test_likelihood => test_likelihood,
    :train_runtimes => train_runtimes,
    :inference_runtimes => inference_runtimes
)

BSON.@save MODEL_WEIGHTS Dict(
    :p => ffjord.p,
)

YAML.write_file(FILENAME, results)
