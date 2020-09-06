using DiffEqFlux, OrdinaryDiffEq, Flux, Distributions, Zygote, LinearAlgebra, RegNeuralODE

# Test convergence on 1D distribution
nn = Chain(Linear(1, 3, tanh), Linear(3, 1, tanh))
tspan = (0.0f0, 10.0f0)
ffjord_test = NFECounterFFJORD(nn, tspan, Tsit5())

data_train = Float32.(rand(Normal(6.0, 0.7), 1, 100))

function loss_adjoint(θ)
    logpx = ffjord_test(data_train, θ)[1]
    loss = -mean(logpx)
end

res1 = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test.p, ADAM(0.01), maxiters = 100)

# Test convergence on 2D distribution
nn = Chain(Linear(2, 8, tanh), Linear(8, 8, tanh), Linear(8, 2))
tspan = (0.f0, 10.f0)
ffjord_test = NFECounterFFJORD(nn, tspan, Tsit5())

data_train = Float32.(rand(MvNormal(ones(2), 3I + zeros(2,2)), 200))

res2 = DiffEqFlux.sciml_train(loss_adjoint, ffjord_test.p, ADAM(0.01), maxiters = 100)