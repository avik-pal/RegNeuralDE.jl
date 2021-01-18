# Classification Network
struct ClassifierNODE{L1,L2,L3,P1,P2,T}
    preode::P1
    node::L2
    postode::P2
    p1::T
    p2::T
    p3::T

    function ClassifierNODE(preode, node, postode)
        p1, re1 = Flux.destructure(preode)
        p2 = node.p
        p3, re3 = Flux.destructure(postode)
        return new{
            typeof(preode),
            typeof(node),
            typeof(postode),
            typeof(re1),
            typeof(re3),
            typeof(p2),
        }(
            re1,
            node,
            re3,
            p1,
            p2,
            p3,
        )
    end
end

Flux.trainable(m::ClassifierNODE) = (m.p1, m.p2, m.p3)

function (m::ClassifierNODE{L1,L2,L3})(
    x,
    p1 = m.p1,
    p2 = m.p2,
    p3 = m.p3;
    node_kwargs...,
) where {L1,L2,L3}
    preode = m.preode(p1)::L1
    x = preode(x)
    x, nfe, sv = m.node(x, p2; node_kwargs...)
    postode = m.postode(p3)::L3
    return postode(x), nfe, sv
end


# Classification Network
struct ClassifierNSDE{L1,L2,L3,P1,P2,T}
    presde::P1
    nsde::L2
    postsde::P2
    p1::T
    p2::T
    p3::T

    function ClassifierNSDE(presde, nsde, postsde)
        p1, re1 = Flux.destructure(presde)
        p2 = nsde.p
        p3, re3 = Flux.destructure(postsde)
        return new{
            typeof(presde),
            typeof(nsde),
            typeof(postsde),
            typeof(re1),
            typeof(re3),
            typeof(p2),
        }(
            re1,
            nsde,
            re3,
            p1,
            p2,
            p3,
        )
    end
end

Flux.trainable(m::ClassifierNSDE) = (m.p1, m.p2, m.p3)

function (m::ClassifierNSDE{L1,L2,L3})(
    x,
    p1 = m.p1,
    p2 = m.p2,
    p3 = m.p3;
    trajectories::Int = 10,
    nsde_kwargs...,
) where {L1,L2,L3}
    # Neural SDE right now works only with CPU Inputs
    x = _expand(x, trajectories) |> track
    presde = m.presde(p1)::L1
    x = presde(x)
    x, nfe1, nfe2, sv = m.nsde(x, p2; nsde_kwargs...)
    postsde = m.postsde(p3)::L3
    z = postsde(x)
    z = mean(reshape(z, size(z, 1), :, trajectories), dims = 3)
    return z, nfe1, nfe2, sv
end

_expand(x::AbstractArray{<:Number, 4}, d::Int) = repeat(x, 1, 1, 1, d)
_expand(x::AbstractArray{<:Number, 2}, d::Int) = repeat(x, 1, d)