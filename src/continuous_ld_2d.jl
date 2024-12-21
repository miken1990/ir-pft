using Parameters
using Distributions
using Random
using LinearAlgebra
using ParticleFilters

@with_kw  mutable struct ContinuousLightDark2D
    correct_r::Float64                              = 30.
    incorrect_r::Float64                            = -30.
    d::Float64                                      = 1.
    beacons                                         = transpose([2.0 4.0 6.0 8.0 9;
                                                                2.0 2.5 3.1 4. 7])
    low_reward_area                                 = transpose([7.0 8.3;
                                                                 6. 9.5])
    low_reward_radius::Float64                      = 3
    movement_cost::Float64                          = -1.
    transition_var::Float64                         = 0.2
    transition_covar::Array{Float64, 2}             = [[transition_var, 0.0]  [0.0, transition_var]]
    observation_var::Float64                        = transition_var * 0.3
    observation_var_base::Array{Float64, 2}         = [[observation_var, 0.0]  [0.0, observation_var]]
    goal_radius::Float64                            = 1.
    goal::Array{Float64, 1}                         = [5, 5]
    init_state_var::Float64                         = observation_var
    initialstate::MvNormal                          = MvNormal([0., 0.], [[init_state_var, 0.0]  [0.0, init_state_var]])
    action_radius::Float64                          = 1
    stay_action::Array{Float64, 1}                  = [0., 0.]
    x_min = -1
    x_max = 11
    y_min = -1
    y_max = 11
    not_in_range_r = -1000
end

###################################### actions ###########################################
stay_action(p::ContinuousLightDark2D)::Array{Float64, 1} = p.stay_action

function random_action(p::ContinuousLightDark2D, rng::MersenneTwister)::Array{Float64, 1}
    angle = rand(rng) * 2π
    return [round(p.action_radius * cos(angle), digits=2), round(p.action_radius * sin(angle), digits=2)]
end

function random_action(p::ContinuousLightDark2D, base::Float64, rng::MersenneTwister)::Array{Float64, 1}
    offset = (rand(rng) - 0.5) * π
    angle = mod(base + offset, 2π)
    return [round(p.action_radius * cos(angle), digits=2), round(p.action_radius * sin(angle), digits=2)]
end

###################################### state transition ##################################
transition(p::ContinuousLightDark2D, s::Array{Float64, 1}, a::Array{Float64, 1}) = MvNormal(s .+ a, p.transition_covar)
motion_likelihood(p::ContinuousLightDark2D, s::Array{Float64, 1}, a::Array{Float64, 1}, sp::Array{Float64, 1}) = pdf(transition(p, s, a), sp)

###################################### observation #######################################
function _closest_beacon_idx(p::ContinuousLightDark2D, s::Array{Float64,1})::Int64
    beacons = p.beacons
    distances = mapslices(norm, beacons .- transpose(s), dims=2)# calculate distances from x to all beacons
    min_idx = argmin(dropdims(distances, dims=2))
    return min_idx
end

# noise multiplier increases linearly as distance from beacon increases
function _noise_func(sp::Array{Float64,1}, beacon::Array{Float64,1})::Float64
    return min(1.0, norm(sp - beacon))
end

function obs_likelihood(p::ContinuousLightDark2D, sp::Array{Float64,1}, 
                        o::Array{Float64,1}, beacon::Vector{Float64})::Float64
    w = p.observation_var * _noise_func(sp, beacon) 
    vv = Matrix(w*Diagonal{Float64}(I, 2)) + p.observation_var_base
    return pdf(MvNormal(sp, vv), o)    
end

# observation distribution will be used to weight samples in particle filter
function observation_d(p::ContinuousLightDark2D, sp::Array{Float64, 1})::Tuple{MvNormal, Vector{Float64}}
    beacon_idx = _closest_beacon_idx(p, sp)
    beacon = p.beacons[beacon_idx, :]
    w = p.observation_var *_noise_func(sp, beacon)
    vv = Matrix(w*Diagonal{Float64}(I, 2))
    return MvNormal(sp, vv), beacon
end

###################################### reward #######################################

function in_range(p::ContinuousLightDark2D, s_new::Array{Float64, 1})::Bool
    return s_new[1] > p.x_min && s_new[1] < p.x_max && s_new[2] > p.y_min && s_new[2] < p.y_max
end 

function reward(p::ContinuousLightDark2D, a::Array{Float64, 1}, s_new::Array{Float64, 1})::Float64
    if norm(s_new - p.goal) < p.goal_radius
        return p.correct_r
    else
        return - 1*norm(s_new-p.goal)
    end
end

#################################### initial state ################################
function initial_state_particles(p::ContinuousLightDark2D, rng::MersenneTwister, n::Int)
    return ParticleCollection([rand(rng, p.initialstate) for i in 1:n])
end

