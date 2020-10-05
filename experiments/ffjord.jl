using DiffEqFlux, OrdinaryDiffEq, Flux, Distributions, Zygote, LinearAlgebra, RegNeuralODE

nn = Chain(Linear(2, 8, tanh), Linear(8, 8, tanh), Linear(8, 2))
tspan = (0.f0, 1.f0)
ffjord = NFECounterFFJORD(nn, tspan, Tsit5())

data_train = Float32.(rand(MvNormal(ones(2), 3I + zeros(2,2)), 100))