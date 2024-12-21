include("likelihood.jl")


# all particles in ba has equal weights since b is resampled
function propagated_belief(p::ContinuousLightDark2D, 
                           b::ParticleCollection{Vector{Float64}}, 
                           a::Array{Float64, 1}, 
                           rng::MersenneTwister)::PropagatedBelief{Float64}
    bp = Vector{typeof(b.particles[1])}(undef, length(b.particles))
    for (i, par) in enumerate(particles(b))
        s_new = rand(rng, transition(p, par, a))
        bp[i] = s_new
    end
    return PropagatedBelief{Float64}(ParticleCollection(bp))
end

# all particles in bp has equal weights
function sample_state_for_obs(bp::ParticleCollection{Vector{Float64}}, 
                              rng::MersenneTwister)::Vector{Float64}
    return rand(rng, bp.particles)
end

# calculates observation weights and performs resampling
function posterior_belief(p::ContinuousLightDark2D, 
    bp::ParticleCollection{Vector{Float64}}, 
    o::Array{Float64,1}, beacon::Vector{Float64}, 
    rng::MersenneTwister)::Belief{Float64}
    w = []
    for par in particles(bp)
        push!(w, obs_likelihood(p, par, o, beacon))
    end
    indexes = wsample(rng, 1:length(w), w/sum(w), length(w))
    return Belief{Float64}(ParticleCollection(particles(bp)[indexes]), 
                                   WeightedParticleBelief(particles(bp), w, sum(w)))
end

function update(p::ContinuousLightDark2D, b::ParticleCollection{Vector{Float64}}, 
                a::Array{Float64, 1}, rng::MersenneTwister)::Tuple{Belief{Float64}, 
                PropagatedBelief{Float64}, Array{Float64,1}, Vector{Float64}, Float64}
    bp = propagated_belief(p, b, a, rng)
    s_o = sample_state_for_obs(bp.bp, rng)
    o_d, beacon = observation_d(p, s_o)
    o = rand(rng, o_d)
    b_new = posterior_belief(p, bp.bp, o, beacon, rng)
    likelihood = calc_likelihood_bp(p, b, bp.bp, a)
    return b_new, bp, o, beacon, likelihood
end

function boers_entropy(p::ContinuousLightDark2D, b_prev::ParticleCollection{Vector{Float64}}, 
                       b_cur::WeightedParticleBelief{Vector{Float64}}, a::Array{Float64, 1}, 
                       o::Array{Float64,1}, beacon::Vector{Float64})::Float64
    weights_prev = weights(b_prev)/weight_sum(b_prev)
    weights_cur = weights(b_cur)/weight_sum(b_cur)
    first_term = 0
    second_term = 0
    third_term = 0
    for (i_cur, (s_cur, w_cur)) in enumerate(zip(particles(b_cur), weights_cur))
        o_l = obs_likelihood(p, s_cur, o, beacon)
        first_term += o_l * weights_prev[i_cur]
        if o_l > eps(10^-50)
            second_term -= w_cur * log(o_l)
        end 
        inner_sum = 0
        for (i_prev, s_prev) in enumerate(particles(b_prev))
            m_l = motion_likelihood(p, s_prev, a, s_cur)
            inner_sum += m_l * weights_prev[i_prev]
        end
        if inner_sum > 0.
            third_term -= w_cur * log(inner_sum)
        end
    end

    first_term = log(first_term)
    res = first_term + second_term + third_term
    return -res
end

function gen_reward(labmda_entropy::Float64, p::ContinuousLightDark2D, b::ParticleCollection{Vector{Float64}}, 
                    b_new::Belief{Float64}, a::Array{Float64, 1}, o::Array{Float64,1}, 
                    beacon::Vector{Float64})::Float64
    s_r = mean([reward(p, s, a) for s in particles(b_new.b)])
    ent_r = labmda_entropy * boers_entropy(p, b, b_new.b_not_resampled, a, o, beacon)
    # println("s_r - $(s_r), ent_r = $(ent_r)")
    return s_r + ent_r
end
