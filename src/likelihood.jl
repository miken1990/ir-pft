using ParticleFilters

function calc_likelihood_bp(p::ContinuousLightDark2D, belief::ParticleCollection{Vector{Float64}},
    bp::ParticleCollection{Vector{Float64}}, a::Array{Float64, 1})::Float64
    num_p = length(belief.particles)
    l_res = 1.
    for i in 1:num_p
        l_res *= motion_likelihood(p, belief.particles[i], a, bp.particles[i])
    end
    return l_res
end