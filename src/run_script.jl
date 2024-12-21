include("types.jl")
include("continuous_ld_2d.jl")
include("updater.jl")
include("likelihood.jl")
include("solve.jl")
include("utils.jl")
include("plot.jl")

using Dates

num_particles_arr = [5, 10, 15, 20]
seed = 15
num_scenarios = 100


problem = ContinuousLightDark2D()
println("num_scenarios = $num_scenarios")
println("seed = $seed")
rng = MersenneTwister(seed)
solver = PFTSolver()
print_every = 5

experiment_stats = Dict{Int64, RuntimeStats}()

for num_particles in num_particles_arr
    println("\n\n num_particles = $num_particles")
    total_g_no_reuse = Vector()
    total_time_no_reuse = Vector()
    b0 = initial_state_particles(problem, rng, num_particles)
    b_start = get_initial_belief(b0)
    o_start = get_dummy_observation(problem)
    beacon_start = get_dummy_beacon(problem)
    for i in 1:num_scenarios
        if i % print_every == 0
            println("scenario no reuse = $i")
        end
        tree = MCTSTree{Float64}()
        b_l = insert_belief_node!(tree, b_start, o_start, beacon_start)
        planner = Planner(solver, 10, problem, tree, rng, 1)
        b_vec, bp_vec, o_vec, a_vec, q_vec, r_vec, n_reused, runtim_vec = solve(planner, b_l)
        g = sum(r_vec)
        runtime = sum(runtim_vec)
        push!(total_g_no_reuse, g)
        push!(total_time_no_reuse, runtime)
    end
    mean_g_no_reuse = mean(total_g_no_reuse)
    std_g_no_reuse = std(total_g_no_reuse)
    mean_runtime_no_reuse = mean(total_time_no_reuse)
    std_runtime_no_reuse = std(total_time_no_reuse)
    println("no reuse: mean G = $mean_g_no_reuse, std G = $std_g_no_reuse, mean time = $mean_runtime_no_reuse, std time = $std_runtime_no_reuse")
    min_n_reuse = 10
    total_g_reuse = Vector()
    total_time_reuse = Vector()
    for i in 1:num_scenarios
        if i % print_every == 0
            println("scenario reuse = $i")
        end
        tree = MCTSTree{Float64}()
        b_l = insert_belief_node!(tree, b_start, o_start, beacon_start)
        planner = Planner(solver, 10, problem, tree, rng, min_n_reuse)
        b_vec, bp_vec, o_vec, a_vec, q_vec, r_vec, n_reused, runtim_vec = solve_with_reuse(planner, b_l)
        g = sum(r_vec)
        runtime = sum(runtim_vec)
        push!(total_g_reuse, g)
        push!(total_time_reuse, runtime)
    end
    speedup = total_time_no_reuse ./ total_time_reuse
    mean_speedup = mean(speedup)
    std_speedup = std(speedup)
    mean_g_reuse = mean(total_g_reuse)
    std_g_reuse = std(total_g_reuse)
    mean_runtime_reuse = mean(total_time_reuse)
    std_runtime_reuse = std(total_time_reuse)
    println("reuse: mean G = $(mean_g_reuse), std G = $(std_g_reuse), mean time = $(mean_runtime_reuse), std time = $(std_runtime_reuse)")
    println("mean speedup = $mean_speedup, std speedup = $std_speedup")
    runtime_stats = RuntimeStats(mean_runtime_no_reuse, std_runtime_no_reuse, mean_runtime_reuse, 
        std_runtime_reuse, mean_g_no_reuse, std_g_no_reuse, mean_g_reuse, std_g_reuse, mean_speedup, std_speedup)
    experiment_stats[num_particles] = runtime_stats
end

plot_runtime(experiment_stats)
plot_accumulated_reward(experiment_stats)
plot_speedup(experiment_stats)

