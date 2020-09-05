# MLP Dynamics conditioned on the Time Step
struct TDChain{T<:Tuple}
    layers::T
    TDChain(xs...) = new{typeof(xs)}(xs)
end

@forward TDChain.layers Base.getindex, Base.length, Base.first, Base.last,
         Base.iterate, Base.lastindex

Flux.functor(::Type{<:TDChain}, c) = c.layers, ls -> TDChain(ls...)

applytdchain(::Tuple{}, x, t) = x
applytdchain(fs::Tuple, x, t) = applytdchain(Base.tail(fs), first(fs)(vcat(x, t)), t)

# Inference for Tracker fails if using the default Flux layers
# https://github.com/FluxML/Tracker.jl/issues/84. As a temporary fix use
# the layers exported from RegNeuralODE
function (c::TDChain)(x::AbstractMatrix, t)
    _t = similar(x, 1, size(x, 2))
    fill!(_t, t)
    return applytdchain(c.layers, x, _t)
end

function (c::TDChain)(x::AbstractMatrix, t::TrackedReal)
    _t = Tracker.collect(ones(eltype(x), 1, size(x, 2))) .* t
    return applytdchain(c.layers, x, _t)
end

Base.getindex(c::TDChain, i::AbstractArray) = TDChain(c.layers[i]...)

Flux.testmode!(m::TDChain, mode = true) = (map(x -> testmode!(x, mode), m.layers); m)
         
function Base.show(io::IO, c::TDChain)
    print(io, "TimeDependentChain(")
    join(io, c.layers, ", ")
    print(io, ")")
end


# Some Common Network Layers

## This layer doesn't have bias. Tracker cause inference issues with Dense
struct Linear{W, S}
    weight::W
    σ::S
end

Linear(W) = Linear(W, identity)

Linear(in::Integer, out::Integer, σ = identity; initW = Flux.glorot_uniform) =
    Linear(initW(out, in), σ)

Flux.@functor Linear

(l::Linear)(x::AbstractArray) = l.σ.(l.weight * x)

function Base.show(io::IO, l::Linear)
    print(io, "Linear(", size(l.weight, 2), ", ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end

## Recognition RNN for mapping from time series data to a latent vector
struct RecognitionRNN{P, Q, R}
    i2h::Dense{P, Q, R}
    h2o::Dense{P, Q, R}
end

Flux.@functor RecognitionRNN

RecognitionRNN(;latent_dim::Int, nhidden::Int, obs_dim::Int) =
    RecognitionRNN(Dense(obs_dim + nhidden, nhidden),
                   Dense(nhidden, latent_dim * 2))

function (rrnn::RecognitionRNN)(x, h)
    h = tanh.(rrnn.i2h(vcat(x, h)))
    return rrnn.h2o(h), h
end