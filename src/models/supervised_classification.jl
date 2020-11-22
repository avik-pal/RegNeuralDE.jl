# Classification Network
struct ClassifierNODE{L1, L2, L3, N, P1, P2, T}
    preode::P1
    node::N
    postode::P2
    p1::T
    p2::T
    p3::T
    
    function ClassifierNODE2(preode, node, postode)
        p1, re1 = Flux.destructure(preode)
        p2 = node.p
        p3, re3 = Flux.destructure(postode)
        return new{typeof(preode), typeof(node), typeof(postode), typeof(node),
                   typeof(re1), typeof(re3), typeof(p2)}(re1, node, re3, p1,
                                                         p2, p3)
    end
end

Flux.trainable(m::ClassifierNODE) = (m.p1, m.p2, m.p3)

function (m::ClassifierNODE{N, P1, P2, T, L1, L2, L3})(
        x, p1 = m.p1, p2 = m.p2, p3 = m.p3) where {N, P1, P2, T, L1, L2, L3}
    preode = m.preode(p1) :: L1
    x = preode(x)
    x, nfe, sv = m.node(x, p2)
    postode = m.postode(p3) :: L3
    return postode(x), nfe, sv
end