include("continuous_ld_2d.jl")

using ParticleFilters
using Parameters
using DataStructures
using NearestNeighbors


mutable struct Belief{T}
    b::ParticleCollection{Vector{T}}
    b_not_resampled::WeightedParticleBelief{Vector{T}}
    mean_val::Vector{T}
    
    function Belief{T}(b::ParticleCollection{Vector{T}}, b_not_resampled::WeightedParticleBelief{Vector{T}}) where T
        new(b, b_not_resampled, mean(particles(b)))
    end
end

mutable struct PropagatedBelief{T}
    bp::ParticleCollection{Vector{T}}
    mean_val::Vector{T}
    
    function PropagatedBelief{T}(bp::ParticleCollection{Vector{T}}) where T
        new(bp,  mean(particles(bp)))
    end
end

mutable struct BeliefTreeNode{T}
    belief::Belief{T}
    label::Int64
    n::Int64
    o::Array{Float64,1}
    beacon::Vector{Float64}
    
    function BeliefTreeNode{T}(belief::Belief{T}, label::Int64, n::Int64,
                               o::Array{Float64,1}, beacon::Vector{Float64}) where T
        new(belief, label, n, o, beacon)
    end
end

mutable struct PropagatedBeliefTreeNode{T}
    prop_belief::PropagatedBelief{T}
    label::Int64
    n::Int64
    q::Float64
    a::Array{Float64, 1}
    likelihood::Float64
    
    function PropagatedBeliefTreeNode{T}(prop_belief::PropagatedBelief{T}, label::Int64, n::Int64,
            q::Float64, a::Array{Float64, 1}, likelihood::Float64) where T
        new(prop_belief, label, n, q, a, likelihood)
    end
end

mutable struct MCTSTree{T}
    b_labels::Dict{Int64, BeliefTreeNode{T}}            # belief label to object
    bp_labels::Dict{Int64, PropagatedBeliefTreeNode{T}} # prop belief label to object
    ba_children::DefaultDict{Tuple{Int64, Array{Float64, 1}}, Vector{Int64}}             # belief label, action to child prop belief labels
    bp_children::DefaultDict{Int64, Vector{Int64}}             # prop belief label to child belief labels
    nba::DefaultDict{Tuple{Int64, Array{Float64, 1}}, Int64}   # belief label, action to nba
    q::DefaultDict{Tuple{Int64, Array{Float64, 1}}, Float64}   # belief label, action to q value
    r::Dict{Int64, Float64}   # belief label to reward
    b_a::DefaultDict{Int64, Vector{Array{Float64, 1}}}  # belief label to actions    
    b_cnt::Int64                                        # number of belief nodes
    bp_cnt::Int64                                       # number of propagated belief nodes
    curr_root::Int64                                    # belief labsel of current root
end

function MCTSTree{T}() where T
    return MCTSTree{T}(
        Dict{Int64, BeliefTreeNode{T}}(), # b_labels
        Dict{Int64, PropagatedBeliefTreeNode{T}}(), # bp_labels
        DefaultDict{Tuple{Int64, Array{Float64, 1}}, Vector{Int64}}(Vector{Int64}), # ba_children
        DefaultDict{Int64, Vector{Int64}}(Vector{Int64}), # bp_children
        DefaultDict{Tuple{Int64, Array{Float64, 1}}, Int64}(() -> 0), # nba
        DefaultDict{Tuple{Int64, Array{Float64, 1}}, Float64}(() -> 0), # q
        Dict{Int64, Float64}(), # r
        DefaultDict{Int64, Vector{Array{Float64, 1}}}(Vector{Array{Float64, 1}}), # b_a
        0, # b_cnt
        0, # bp_cnt
        0  # cur_root
    )
end


function insert_belief_node!(tree::MCTSTree{T}, belief::Belief{T}, 
    o::Array{Float64,1}, beacon::Vector{Float64})::Int64 where T
    b_tree_node = BeliefTreeNode{T}(belief, tree.b_cnt, 0, o, beacon)
    b_label = tree.b_cnt
    tree.b_labels[b_label] = b_tree_node
    tree.b_cnt += 1
    return b_label
end

function insert_belief_node!(tree::MCTSTree{T}, belief::Belief{T}, 
    o::Array{Float64,1}, beacon::Vector{Float64}, n::Int64)::Int64 where T
    b_tree_node = BeliefTreeNode{T}(belief, tree.b_cnt, 0, o, beacon)
    b_tree_node.n = n
    b_label = tree.b_cnt
    tree.b_labels[b_label] = b_tree_node
    tree.b_cnt += 1
    return b_label
end

function insert_prop_belief_node!(tree::MCTSTree{T}, bp::PropagatedBelief{T}, 
    b_label::Int64, b_child_label::Int64, q::Float64, a::Array{Float64, 1}, 
        likelihood::Float64) where T
    bp_l = tree.bp_cnt
    bp_tree_node = PropagatedBeliefTreeNode{T}(bp, bp_l, 0, q, a, likelihood)
    tree.bp_labels[bp_l] = bp_tree_node
    push!(tree.ba_children[(b_label, a)], bp_l)
    push!(tree.bp_children[bp_l], b_child_label)
    tree.bp_cnt += 1
    return bp_l
end

function insert_prop_belief_node!(tree::MCTSTree{T}, bp::PropagatedBelief{T}, 
    b_label::Int64, b_child_label::Int64, q::Float64, a::Array{Float64, 1}, 
        likelihood::Float64, n::Int64) where T
    bp_l = tree.bp_cnt
    bp_tree_node = PropagatedBeliefTreeNode{T}(bp, bp_l, 0, q, a, likelihood)
    bp_tree_node.n = n
    tree.bp_labels[bp_l] = bp_tree_node
    push!(tree.ba_children[(b_label, a)], bp_l)
    push!(tree.bp_children[bp_l], b_child_label)
    tree.bp_cnt += 1
    return bp_l
end

# solvers - PFT and IR-PFT

mutable struct  PFTSolver
    depth::Int64
    exploration_c::Float64
    n_iterations::Int
    k_s::Float64
    alpha_s::Float64
    k_a::Float64
    alpha_a::Float64
    gamma::Float64
    log::Bool
    lambda_entropy::Float64
    num_reused_simulations::Int64
    function PFTSolver(
        depth::Int=20,
        # depth::Int=2,
        exploration_c::Float64=0.1,
        n_iterations::Int=1000,
        # n_iterations::Int=10,
        k_s::Float64=1.0,
        alpha_s::Float64=0.1,
        k_a::Float64=1.0,
        alpha_a::Float64=0.1,
        gamma::Float64=0.95,
        log::Bool=true,
        lambda_entropy::Float64=10.0,
        )
        new(depth, exploration_c, n_iterations, k_s, alpha_s, k_a, alpha_a, 
            gamma, log, lambda_entropy, 0)
    end
end

# Planner
mutable struct Planner
    solver::PFTSolver
    n_steps::Int64
    problem::ContinuousLightDark2D
    tree::MCTSTree
    rng::MersenneTwister
    min_n_reuse::Int64
end


mutable struct KDTreeLabels
    tree::KDTree
    uids::Vector{Int64}
    mean::Vector{Array{Float64, 1}}
    function KDTreeLabels(tree::KDTree, uids::Vector{Int64}, mean::Vector{Array{Float64, 1}})
        new(tree, uids, mean)
    end
end

function build_kd_tree_labels(labels::Vector{Tuple{Int64, Array{Float64, 1}}})
    label_mean = [label[2] for label in labels]
    uids = [label[1] for label in labels]
    tree = KDTree(hcat(label_mean...))
    return KDTreeLabels(tree, uids, label_mean)
end

function nearest_neighbor_label(kd_tree_labels::KDTreeLabels, query_point::Array{Float64, 1})::Int64
    idxs, _ = knn(kd_tree_labels.tree, query_point, 1, true)
    return kd_tree_labels.uids[idxs[1]]
end

##################### reuse ########################
mutable struct Reuse
    reuse_label_mean::KDTreeLabels
    num_reused_b_l_a::DefaultDict{Array{Float64, 1}, Int64}
    bp_l_to_parent_b_l_a::Dict{Int64, Tuple{Int64, Array{Float64, 1}}}
    bl_a_bp_l_to_likelihood::Dict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64}
    bl_a_bp_l_to_q::DefaultDict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64}
    denom_likelihood::DefaultDict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64} #b_l, a, bp_l to mis denom
    numerator_likelihood::DefaultDict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64} #b_l, a, bp_l to mis num
    horizon_filled_bpl::DefaultDict{Int64, Bool}
    function Reuse(reuse_label_mean::KDTreeLabels, 
                bp_l_to_parent_b_l_a::Dict{Int64, Tuple{Int64, Array{Float64, 1}}})
        new(reuse_label_mean, 
            DefaultDict{Array{Float64, 1}, Int64}(() -> 0),
            bp_l_to_parent_b_l_a, 
            Dict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64}(),
            DefaultDict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64}(() -> 0),
            DefaultDict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64}(() -> 0),
            DefaultDict{Tuple{Int64, Array{Float64, 1}, Int64}, Float64}(() -> 0),
            DefaultDict{Int64, Bool}(() -> false)
        )
    end
end

function should_reuse(reuse::Reuse, b_l::Int64, a::Array{Float64, 1}, num_nodes::Int64)::Bool
    return num_nodes / 2 > reuse.num_reused_b_l_a[a]
end

################### runtime stats ####################
mutable struct RuntimeStats
    mean_runtime_no_reuse::Float64
    std_runtime_no_reuse::Float64
    mean_runtime_reuse::Float64
    std_runtime_reuse::Float64
    mean_g_no_reuse::Float64
    std_g_no_reuse::Float64
    mean_g_reuse::Float64
    std_g_reuse::Float64
    mean_speedup::Float64
    std_speedup::Float64
    function RuntimeStats(mean_runtime_no_reuse::Float64, std_runtime_no_reuse::Float64,
        mean_runtime_reuse::Float64, std_runtime_reuse::Float64, mean_g_no_reuse::Float64,
        std_g_no_reuse::Float64, mean_g_reuse::Float64, std_g_reuse::Float64, mean_speedup::Float64,
        std_speedup::Float64)
        new(mean_runtime_no_reuse, std_runtime_no_reuse, mean_runtime_reuse, std_runtime_reuse,
            mean_g_no_reuse, std_g_no_reuse, mean_g_reuse, std_g_reuse, mean_speedup, std_speedup)
    end
end