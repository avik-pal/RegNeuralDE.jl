# Classification Network
struct ClassifierNODE{N, P1, P2, T}
    preode::P1
    node::N
    postode::P2
    p1::T
    p2::T
    p3::T
    
    function ClassifierNODE(preode, node, postode)
        p1, re1 = Flux.destructure(preode)
        p2 = node.p
        p3, re3 = Flux.destructure(postode)
        return new{typeof(node), typeof(re1), typeof(re3),
                   typeof(p2)}(re1, node, re3, p1, p2, p3)
    end
end

Flux.trainable(m::ClassifierNODE) = (m.p1, m.p2, m.p3)

function (m::ClassifierNODE)(x, p1 = m.p1, p2 = m.p2, p3 = m.p3)
    x = m.preode(p1)(x)
    x = m.node(x, p2)
    return m.postode(p3)(x)
end

function (m::ClassifierNODE{T})(x, p1 = m.p1, p2 = m.p2,
                                p3 = m.p3) where T<:NFECounterCallbackNeuralODE 
    x = m.preode(p1)(x)
    x, sv = m.node(x, p2)
    return m.postode(p3)(x), sv
end