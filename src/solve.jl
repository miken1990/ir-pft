include("types.jl")
include("updater.jl")

using NearestNeighbors


function fill_horizon_posterior(tree::MCTSTree, b_l::Int64,
    p::ContinuousLightDark2D, solver::PFTSolver, rng::MersenneTwister)::Float64
    actions = tree.b_a[b_l]
    # b is leaf
    if length(actions) == 0
        cur_b = tree.b_labels[b_l]
        a = get_next_action(tree, p, solver, b_l, rng)
        b_new, bp, o, beacon, likelihood = update(p, cur_b.belief.b,
                                                    a, rng)
        new_b_l = insert_belief_node!(tree, b_new, o, beacon, 1)
        new_bp_l = insert_prop_belief_node!(tree, bp, b_l, new_b_l, 0., a, likelihood)
        r = gen_reward(solver.lambda_entropy, problem, 
                       cur_b.belief.b, b_new, a, o, beacon)
        tree.r[new_b_l] = r
        return r
    end
    q_b = 0
    num_prop = 0
    # b is not leaf, iterate over children recursively
    for a in actions
        for bp_l in tree.ba_children[(b_l, a)]
            q_b += _fill_horizon_propagated(tree, bp_l, p, solver, rng)
            num_prop += 1
        end
    end
    return solver.gamma * q_b / num_prop   
end

function _fill_horizon_propagated(tree::MCTSTree, bp_l::Int64,
                            p::ContinuousLightDark2D, solver::PFTSolver, rng::MersenneTwister)::Float64
    q_new = 0
    bp = tree.bp_labels[bp_l]
    bp_children_l = tree.bp_children[bp_l]
    for b_next_l in bp_children_l
        q_new = q_new + fill_horizon_posterior(tree, b_next_l, p, solver, rng)
    end
    bp.q = q_new / length(bp_children_l)
    return bp.q
end

function get_next_action(tree::MCTSTree, p::ContinuousLightDark2D, 
                        solver::PFTSolver, cur_b_l::Int64, 
                        rng::MersenneTwister)::Array{Float64, 1}
    b_cur_node = tree.b_labels[cur_b_l]
    num_actions = length(tree.b_a[cur_b_l])
    # we have budget - sample random action
    a_th = solver.k_a * (b_cur_node.n ^ solver.alpha_a)
    if num_actions <= a_th
        dir_vec = p.goal - b_cur_node.belief.mean_val
        base_angle = atan(dir_vec[2], dir_vec[1])
        a = random_action(p, base_angle, rng)
        push!(tree.b_a[cur_b_l], a)
        return a
    # no budget -  return the action with largest UCB value
    else
        mab_q = -Inf
        best_a = nothing
        actions = tree.b_a[cur_b_l]
        for a in actions
            # q will be already updated with the correct value
            q = tree.q[(cur_b_l, a)]
            nba = tree.nba[(cur_b_l, a)]
            c = solver.exploration_c * sqrt(log(b_cur_node.n) / nba)
            if q + c > mab_q
                mab_q = q + c
                best_a = a
            end
        end
        return best_a
    end
end

# simulate function for non-root nodes
function simulate(planner::Planner, d::Int64, prev_b_l::Int64, 
                  cur_b_l::Int64, prev_a::Array{Float64, 1})::Float64
    # reward caluclation
    r = 0
    pft_solver = planner.solver
    cur_b = planner.tree.b_labels[cur_b_l]
    # check if cached
    if haskey(planner.tree.r, cur_b_l)
        r = planner.tree.r[cur_b_l]
    # calculate and cache
    else
        prev_b = planner.tree.b_labels[prev_b_l]
        r = gen_reward(pft_solver.lambda_entropy, planner.problem, 
                       prev_b.belief.b, cur_b.belief, prev_a, cur_b.o, cur_b.beacon)
        planner.tree.r[cur_b_l] = r

    end
    # reached max depth
    if d == 0
        cur_b.n += 1
        return r
    end

    a = get_next_action(planner.tree, planner.problem, 
        planner.solver, cur_b_l, planner.rng)
    s_threshold = max(planner.solver.k_s * (cur_b.n ^ planner.solver.alpha_s), 0.1)
    # we have exactly same number of propagated and posterior beliefs
    num_children = length(planner.tree.ba_children[(cur_b_l, a)])
    total = 0
    new_bp_l = -1
    # we have budget for new next belief
    if num_children < s_threshold
        b_new, bp, o, beacon, likelihood = update(planner.problem, cur_b.belief.b,
                                                  a, planner.rng)
        new_b_l = insert_belief_node!(planner.tree, b_new, o, beacon)
        new_bp_l = insert_prop_belief_node!(planner.tree, bp, cur_b_l, new_b_l, 0., a, likelihood)
        total = r + planner.solver.gamma * simulate(planner, d-1, cur_b_l, new_b_l, a)
    # sample uniformly from existing children
    else
        new_bp_l = sample(planner.rng, planner.tree.ba_children[(cur_b_l, a)])
        new_b_l = planner.tree.bp_children[new_bp_l][1]
        total = r + planner.solver.gamma * simulate(planner, d-1, cur_b_l, new_b_l, a)
    end
    # update UCT values
    new_bp = planner.tree.bp_labels[new_bp_l]
    new_bp.n += 1
    new_bp.q = new_bp.q + (total - new_bp.q) / new_bp.n 

    cur_b.n += 1
    planner.tree.nba[(cur_b_l, a)] += 1
    planner.tree.q[(cur_b_l, a)] = planner.tree.q[(cur_b_l, a)] + (total - planner.tree.q[(cur_b_l, a)]) / planner.tree.nba[(cur_b_l, a)]
    return total
end

# simulate function for root node
function simulate_root(planner::Planner, d::Int64,  cur_b_l::Int64)::Int64
    cur_b = planner.tree.b_labels[cur_b_l]
    a = get_next_action(planner.tree, planner.problem, 
        planner.solver, cur_b_l, planner.rng)
    s_threshold = max(planner.solver.k_s * (cur_b.n ^ planner.solver.alpha_s), 0.1)
    # we have exactly same number of propagated and posterior beliefs
    num_children = length(planner.tree.ba_children[(cur_b_l, a)])
    total = 0
    new_bp_l = -1
    # we have budget for new next belief
    if num_children < s_threshold
        b_new, bp, o, beacon, likelihood = update(planner.problem, cur_b.belief.b,
                                                a, planner.rng)
        new_b_l = insert_belief_node!(planner.tree, b_new, o, beacon)
        new_bp_l = insert_prop_belief_node!(planner.tree, bp, cur_b_l, new_b_l, 0., a, likelihood)
        total = planner.solver.gamma * simulate(planner, d-1, cur_b_l, new_b_l, a)
    # sample uniformly from existing children
    else
        new_bp_l = sample(planner.rng, planner.tree.ba_children[(cur_b_l, a)])
        new_b_l = planner.tree.bp_children[new_bp_l][1]
        total = planner.solver.gamma * simulate(planner, d-1, cur_b_l, new_b_l, a)
    end
    # update UCT values
    new_bp = planner.tree.bp_labels[new_bp_l]
    new_bp.n += 1
    new_bp.q = new_bp.q + (total - new_bp.q) / new_bp.n 
    cur_b.n += 1
    planner.tree.nba[(cur_b_l, a)] = planner.tree.nba[(cur_b_l, a)]  + 1
    planner.tree.q[(cur_b_l, a)] = planner.tree.q[(cur_b_l, a)] + (total - planner.tree.q[(cur_b_l, a)]) / planner.tree.nba[(cur_b_l, a)]
    return 1
end

function get_best_action_and_q(planner::Planner , b_l::Int64)
    actions = planner.tree.b_a[b_l]
    max_q = -Inf
    best_a = nothing
    for a in actions
        # q will be already updated with the correct value
        q = planner.tree.q[(b_l, a)]
        if q > max_q
            max_q = q
            best_a = a
        end
    end
    return best_a, max_q
end

function plan(planner::Planner, b_l::Int64) 
    i = 1
    n_reused = 0
    while i < planner.solver.n_iterations
        num_reused = simulate_root(planner, planner.solver.depth, b_l)
        i += num_reused
        n_reused += num_reused
    end
    return n_reused
end

function solve(planner::Planner, b_l::Int64)
    cur_b = planner.tree.b_labels[b_l] 
    b_vec = Vector()
    push!(b_vec, cur_b.belief)
    bp_vec = Vector()
    o_vec = Vector()
    a_vec = Vector()
    q_vec = Vector()
    r_vec = Vector()
    n_reused_vec = Vector()
    runtim_vec = Vector()
    for i in 1:planner.n_steps
        start_simulate_time = Dates.now()
        num_reused = plan(planner, b_l)
        end_simulate_time = Dates.now()
        t_sim_seconds = get_dt_in_seconds(start_simulate_time, end_simulate_time)
        best_a, max_q = get_best_action_and_q(planner, b_l)
        cur_b = planner.tree.b_labels[b_l] 
        b_new, bp, o, beacon, likelihood = update(planner.problem, cur_b.belief.b,
                                                  best_a, planner.rng)
        r = gen_reward(planner.solver.lambda_entropy, planner.problem, cur_b.belief.b,
                      b_new, best_a, o, beacon)
        push!(b_vec, b_new)
        push!(bp_vec, bp)
        push!(o_vec, o)
        push!(a_vec, best_a)
        push!(q_vec, max_q)
        push!(r_vec, r)
        push!(n_reused_vec, num_reused)
        push!(runtim_vec, t_sim_seconds)
        b_l = insert_belief_node!(planner.tree, b_new, o, beacon)
    end
    return b_vec, bp_vec, o_vec, a_vec, q_vec, r_vec, n_reused_vec, runtim_vec
end

function get_reuse_candidates(planner::Planner, b_l::Int64)::Reuse
    reuse_vec = Vector{Tuple{Int64, Array{Float64, 1}}}()
    bp_l_to_parent_b_l_a = Dict{Int64, Tuple{Int64, Array{Float64, 1}}}()
    for a in planner.tree.b_a[b_l]
        ba_children_l = planner.tree.ba_children[(b_l, a)]
        for ba_l in ba_children_l
            bp_children_l = planner.tree.bp_children[ba_l]
            for b_next_l in bp_children_l
                actions = planner.tree.b_a[b_next_l]
                for act in actions
                    bp_reuse_l = planner.tree.ba_children[(b_next_l, act)]
                    for bp_l in bp_reuse_l
                        bp = planner.tree.bp_labels[bp_l]
                        if bp.n > planner.min_n_reuse
                            push!(reuse_vec, (bp_l, bp.prop_belief.mean_val))
                            bp_l_to_parent_b_l_a[bp_l] = (b_next_l, act)
                        end
                    end
                end
            end
        end
    end
    
    kdtree = build_kd_tree_labels(reuse_vec)
    return Reuse(kdtree, bp_l_to_parent_b_l_a)
end

# incremental update implementation
function mis_update(problem::ContinuousLightDark2D, reuse::Reuse, tree::MCTSTree, cur_b_l::Int64, 
                    a::Array{Float64, 1}, new_bp_l::Int64, 
                    n_new::Int64, q_new::Float64, n_old::Int64, 
                    q_old::Float64)::Float64
    q = 0
    # incremental updates
    for bp_l in tree.ba_children[(cur_b_l, a)]
        # denom update is common to all distributions
        b_prev, a_prev = reuse.bp_l_to_parent_b_l_a[bp_l]
        if !haskey(reuse.bl_a_bp_l_to_likelihood, (b_prev, a_prev, bp_l))
            reuse.bl_a_bp_l_to_likelihood[(b_prev, a_prev, bp_l)] = 
                calc_likelihood_bp(problem, tree.b_labels[b_prev].belief.b,
                                    tree.bp_labels[bp_l].prop_belief.bp, a_prev)
        end
        reuse.denom_likelihood[(cur_b_l, a, bp_l)] = reuse.denom_likelihood[(cur_b_l, a, bp_l)] + 
        n_new * reuse.bl_a_bp_l_to_likelihood[(b_prev, a_prev, bp_l)]
        bp = tree.bp_labels[bp_l]
        # other updates are specific to new_bp_l distribution
        if bp_l == new_bp_l
            if !haskey(reuse.bl_a_bp_l_to_likelihood, (cur_b_l, a, bp_l))
                reuse.bl_a_bp_l_to_likelihood[(cur_b_l, a, bp_l)] = 
                    calc_likelihood_bp(problem, tree.b_labels[cur_b_l].belief.b,
                                        tree.bp_labels[bp_l].prop_belief.bp, a)
            end
            reuse.numerator_likelihood[(cur_b_l, a, bp_l)] = reuse.numerator_likelihood[(cur_b_l, a, bp_l)] + 
                n_new * reuse.bl_a_bp_l_to_likelihood[(cur_b_l, a, bp_l)]
            bp.q = (1 / (n_new + n_old)) * (n_old * q_old + n_new * q_new)
            bp.n += n_new
        end
        q += (reuse.numerator_likelihood[(cur_b_l, a, bp_l)]) * bp.q / 
            (reuse.denom_likelihood[(cur_b_l, a, bp_l)])
    end
    return q
end

# simulate function with reuse
function simulate_root_with_reuse(planner::Planner, d::Int64,  cur_b_l::Int64, reuse::Reuse)::Int64
    cur_b = planner.tree.b_labels[cur_b_l]
    a = get_next_action(planner.tree, planner.problem, 
        planner.solver, cur_b_l, planner.rng)
    s_threshold = max(planner.solver.k_s * (cur_b.n ^ planner.solver.alpha_s), 0.1)
    # we have exactly same number of propagated and posterior beliefs
    num_children = length(planner.tree.ba_children[(cur_b_l, a)])
    total = 0
    new_bp_l = -1
    # we have budget for new next belief
    if num_children < s_threshold
        # add reuse 
        if should_reuse(reuse, cur_b_l, a, length(planner.tree.ba_children[(cur_b_l, a)]))
            new_bp_l = nearest_neighbor_label(reuse.reuse_label_mean, cur_b.belief.mean_val + a)
            if !reuse.horizon_filled_bpl[new_bp_l]
                _fill_horizon_propagated(planner.tree, new_bp_l, planner.problem, 
                                         planner.solver, planner.rng)
                reuse.horizon_filled_bpl[new_bp_l] = true
            end
            # fix initial reward
            new_b_l = planner.tree.bp_children[new_bp_l][1]
            new_b = planner.tree.b_labels[new_b_l]
            r = gen_reward(planner.solver.lambda_entropy, planner.problem, 
                           cur_b.belief.b, new_b.belief, a, cur_b.o, cur_b.beacon)
            new_bp = planner.tree.bp_labels[new_bp_l]
            new_bp.q = new_bp.q + r - planner.tree.r[new_b_l]
            planner.tree.r[new_b_l] = r
            cur_b.n += new_bp.n
            planner.tree.nba[(cur_b_l, a)] += new_bp.n
            # connect new_bp to cur_b, a
            push!(planner.tree.ba_children[(cur_b_l, a)], new_bp_l)
            # calculate q using MIS update
            planner.tree.q[(cur_b_l, a)] = mis_update(planner.problem, reuse, 
                planner.tree, cur_b_l, a, new_bp_l, new_bp.n, new_bp.q, 0, 0.)
            return new_bp.n
        else
            b_new, bp, o, beacon, likelihood = update(planner.problem, cur_b.belief.b,
            a, planner.rng)
            new_b_l = insert_belief_node!(planner.tree, b_new, o, beacon)
            new_bp_l = insert_prop_belief_node!(planner.tree, bp, cur_b_l, new_b_l, 0., a, likelihood)
            total = planner.solver.gamma * simulate(planner, d-1, cur_b_l, new_b_l, a)
            reuse.bp_l_to_parent_b_l_a[new_bp_l] = (cur_b_l, a)
        end
    # sample uniformly from existing children
    else
        new_bp_l = sample(planner.rng, planner.tree.ba_children[(cur_b_l, a)])
        new_b_l = planner.tree.bp_children[new_bp_l][1]
        total = planner.solver.gamma * simulate(planner, d-1, cur_b_l, new_b_l, a)
    end    
    cur_b.n += 1
    n_old = planner.tree.nba[(cur_b_l, a)] = planner.tree.nba[(cur_b_l, a)]
    q_old = planner.tree.q[(cur_b_l, a)]
    planner.tree.q[(cur_b_l, a)] = mis_update(planner.problem,reuse, 
        planner.tree, cur_b_l, a, new_bp_l, 1, total, n_old, q_old)
    planner.tree.nba[(cur_b_l, a)] = planner.tree.nba[(cur_b_l, a)]  + 1
    return 1
end

function plan_with_reuse(planner::Planner, b_l::Int64, reuse::Reuse) 
    i = 1
    n_reused = 0
    while i < planner.solver.n_iterations
        num_reused = simulate_root_with_reuse(planner, planner.solver.depth, b_l, reuse)
        i += num_reused
        n_reused += num_reused
    end
    return n_reused
end

function solve_with_reuse(planner::Planner, b_l::Int64)
    cur_b = planner.tree.b_labels[b_l] 
    b_vec = Vector()
    push!(b_vec, cur_b.belief)
    bp_vec = Vector()
    o_vec = Vector()
    a_vec = Vector()
    q_vec = Vector()
    r_vec = Vector()
    n_reused_vec = Vector()
    runtim_vec = Vector()
    reuse_candidates = nothing
    for i in 1:planner.n_steps
        start_simulate_time = Dates.now()
        num_reused = -1
        if i > 1
            num_reused = plan_with_reuse(planner, b_l, reuse_candidates)
        else
            num_reused = plan(planner, b_l)
        end
        end_simulate_time = Dates.now()
        t_sim_seconds = get_dt_in_seconds(start_simulate_time, end_simulate_time)
        best_a, max_q = get_best_action_and_q(planner, b_l)
        cur_b = planner.tree.b_labels[b_l] 
        b_new, bp, o, beacon, likelihood = update(planner.problem, cur_b.belief.b,
                                                  best_a, planner.rng)
        r = gen_reward(planner.solver.lambda_entropy, planner.problem, cur_b.belief.b,
                      b_new, best_a, o, beacon)
        push!(b_vec, b_new)
        push!(bp_vec, bp)
        push!(o_vec, o)
        push!(a_vec, best_a)
        push!(q_vec, max_q)
        push!(r_vec, r)
        push!(n_reused_vec, num_reused)
        push!(runtim_vec, t_sim_seconds)
        # update reuse candidates
        reuse_candidates = get_reuse_candidates(planner, b_l)
        b_l = insert_belief_node!(planner.tree, b_new, o, beacon)
    end
    return b_vec, bp_vec, o_vec, a_vec, q_vec, r_vec, n_reused_vec, runtim_vec
end