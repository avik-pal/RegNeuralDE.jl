using RegNeuralODE, OrdinaryDiffEq, Flux, CUDA, Tracker
CUDA.allowscalar(false)

mlp_dynamics = TDChain(Dense(3, 10, CUDA.tanh), Dense(11, 2)) |> track |> gpu

x_ = rand(2, 1) |> gpu

# Unregularized Neural ODE
node_unreg = TrackedNeuralODE(
    mlp_dynamics,
    [0.0f0, 1.0f0],
    true,
    false,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-8,
    abstol = 1.4f-8,
    save_start = false,
)

## Forward Pass
@code_warntype node_unreg(x_ |> track)

## Backward Pass
@code_warntype Tracker.gradient(p -> sum(node_unreg(x_ |> track, p)[1]), node_unreg.p)

# Regularized Neural ODE -- Error Estimates
node_errreg = TrackedNeuralODE(
    mlp_dynamics,
    [0.0f0, 1.0f0],
    true,
    true,
    Tsit5(),
    save_everystep = false,
    reltol = 1.4f-8,
    abstol = 1.4f-8,
    save_start = false,
)

## Forward Pass
@code_warntype node_errreg(
    x_ |> track;
    func = (u, t, integrator) -> integrator.EEst * integrator.dt,
)

## Backward Pass
@code_warntype Tracker.gradient(
    p -> begin
        res, _, sv = node_errreg(
            x_ |> track,
            p;
            func = (u, t, integrator) -> integrator.EEst * integrator.dt,
        )
        sum(res) + sum(sv.saveval)
    end,
    node_errreg.p,
)

# Regularized Neural ODE -- Stiffness Estimates
node_stiffreg = TrackedNeuralODE(
    mlp_dynamics,
    [0.0f0, 1.0f0],
    true,
    true,
    AutoTsit5(Tsit5()),
    save_everystep = false,
    reltol = 1.4f-8,
    abstol = 1.4f-8,
    save_start = false,
)

## Forward Pass
@code_warntype node_stiffreg(
    x_ |> track;
    func = (u, t, integrator) -> abs(integrator.eigen_est * integrator.dt),
)

## Backward Pass
@code_warntype

Tracker.gradient(
    p -> begin
        res, _, sv = node_stiffreg(
            x_ |> track,
            p;
            func = (u, t, integrator) -> abs(integrator.eigen_est * integrator.dt),
        )
        sum(res) + sum(filter(x -> !iszero(x), sv.saveval))
    end,
    node_stiffreg.p,
)
