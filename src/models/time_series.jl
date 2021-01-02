struct LatentTimeSeriesModel{R,E,N,D,R1,E1,D1,T}
    rnn::R1
    enc::E1
    node::N
    dec::D1
    p1::T
    p2::T
    p3::T
    p4::T

    function LatentTimeSeriesModel(rnn, enc, node, dec)
        p1, re1 = Flux.destructure(rnn)
        p2, re2 = Flux.destructure(enc)
        p3 = node.p
        p4, re3 = Flux.destructure(dec)
        return new{
            typeof(rnn),
            typeof(enc),
            typeof(node),
            typeof(dec),
            typeof(re1),
            typeof(re2),
            typeof(re3),
            typeof(p3),
        }(
            re1,
            re2,
            node,
            re3,
            p1,
            p2,
            p3,
            p4,
        )
    end
end

Flux.trainable(m::LatentTimeSeriesModel) = (m.p1, m.p2, m.p3, m.p4)

function (m::LatentTimeSeriesModel{R,E,N,D})(
    x::AbstractArray{T,3},
    p1 = m.p1,
    p2 = m.p2,
    p3 = m.p3,
    p4 = m.p4;
    node_kwargs...,
) where {R,E,N,D,T}
    rnn = m.rnn(p1)::R
    out = rnn(x)

    enc = m.enc(p2)::E
    out = enc(out)

    latent_dim = size(out, 1) ÷ 2
    μ₀ = out[1:latent_dim, :]
    logσ² = out[latent_dim+1:end, :]

    sample = CUDA.randn(size(μ₀, 1), size(μ₀, 2))::CuArray{Float32,2}
    z₀ = sample .* exp.(logσ² / 2) .+ μ₀

    res, nfe, sv = m.node(z₀, p3; node_kwargs...)

    dec = m.dec(p4)::D

    res = reshape(res, size(res, 1), :)
    result = dec(res)
    result = reshape(result, size(result, 1), :, size(x, 3))

    return result, μ₀, logσ², nfe, sv
end
