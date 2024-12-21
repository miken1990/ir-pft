function get_dt_in_seconds(start_time, end_time)::Float64
    it_delta_time = end_time-start_time
    t_s = it_delta_time.value / 1000
    return t_s
end

function get_initial_belief(b0)
    num_particles = length(particles(b0))
    w = fill(1 / num_particles, num_particles)
    return Belief{Float64}(ParticleCollection(particles(b0)), 
        WeightedParticleBelief(particles(b0), w, sum(w)))
end

function get_dummy_beacon(problem::ContinuousLightDark2D)
    return Array{Float64, 1}([1., 1.,])
end

function get_dummy_observation(problem::ContinuousLightDark2D)
    return Array{Float64, 1}([1., 1.,])
end