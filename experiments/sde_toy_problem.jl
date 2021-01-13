using Plots, Statistics
using Flux, DiffEqFlux, StochasticDiffEq, CUDA, RegNeuralODE, Tracker
using BSON
using Tracker: data

dataset = BSON.load("data/sde_demo.bson")
sde_data = dataset[:sde_data]
sde_data_vars = dataset[:sde_data_vars]

u0 = reshape(Float32[2.0; 0.0], :, 1)
datasize = 30
tspan = [0.0f0, 1.0f0]
tsteps = range(tspan[1], tspan[2], length = datasize) |> track

drift_dudt = Chain(Dense(2, 10, tanh), Dense(10, 2)) |> track
diffusion_dudt = Dense(2, 2) |> track

REGULARIZE = false

neuralsde = TrackedNeuralDSDE(
    drift_dudt,
    diffusion_dudt,
    tspan,
    REGULARIZE,
    SOSRI(),
    saveat = tsteps,
    reltol = 1f-1,  # Increasing tolerance leads to StackOverflow
    abstol = 1f-1,
)

function loss_function(u0, p, i)
    sol, nfe1, nfe2, sv = neuralsde(u0, p)
    means, vars = mean(sol; dims = 3), var(sol; dims = 3)
    l2_means = mean(abs2, sde_data .- means)
    l2_vars = mean(abs2, sde_data_vars .- vars)
    reg = REGULARIZE ? 50 * mean(sv.saveval) : 0.0f0
    loss = l2_means + l2_vars + reg
    @show i, data(loss), data(l2_means), data(l2_vars), data(reg), nfe1, nfe2
    return loss
end

u0_ = repeat(u0, 1, 100)
loss_function(u0_ |> track, neuralsde.p, 0)
Tracker.gradient(p -> loss_function(u0_ |> track, p, 0), neuralsde.p)
opt = ADAM(0.025)
ps = neuralsde.p

for iter in 1:250
    gs = Tracker.gradient(p -> loss_function(u0_ |> track, p, iter), ps)[1]
    update_parameters!((ps,), (gs,), opt)
end

preds = neuralsde(u0_ |> track)[1] |> untrack
means, vars = mean(preds; dims = 3), var(preds, dims = 3)

plot(tsteps |> untrack, means[1, :, 1], ribbon = vars[1, :, 1])
plot!(tsteps |> untrack, means[2, :, 1], ribbon = vars[2, :, 1])
scatter!(
    tsteps |> untrack,
    sde_data[1, :],
    ribbon = sde_data_vars[1, :],
    label = "data dim 1",
    color = :blue,
    fillalpha = 0.2,
)
scatter!(
    tsteps |> untrack,
    sde_data[2, :],
    ribbon = sde_data_vars[2, :],
    label = "data dim 2",
    color = :red,
    fillalpha = 0.2,
)
