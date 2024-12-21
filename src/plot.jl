include("continuous_ld_2d.jl")
include("types.jl")

using Plots


function plot_belief_traj(problem::ContinuousLightDark2D, belief_traj)
    start_idx = 1
    belief_traj=belief_traj[start_idx:end]
    # plot beacons
    x_coord = []
    y_coord = []
    for beacon in eachrow(problem.beacons)
        push!(x_coord, beacon[1])
        push!(y_coord, beacon[2])
    end
    p = scatter(
        x_coord, y_coord,
        marker=(:utriangle, 8),
        markercolor=RGB(173 / 255, 216 / 255, 255 / 255),
        label="Beacons",
        background_color=RGB(0.3, 0.3, 0.3),
        grid=false,
        axis=false,
        legend=:bottomright,
        legendfont = font(20),
        aspect_ratio=:equal,
        x_ticks=nothing,
        y_ticks=nothing
    )

    # plot mean belief traj
    x_coord = []
    y_coord = []
    for b in belief_traj
        push!(x_coord, b.mean_val[1])
        push!(y_coord, b.mean_val[2])
    end
    plot!(
        x_coord, y_coord,
        marker=(:circle, 5),
        markercolor=RGB(255 / 255, 255 / 255, 255 / 255),
        label="Mean Belief"
    )

    # plot belief points
    for (i, b) in enumerate(belief_traj)
        x_coord = []
        y_coord = []
        for s in particles(b.b)
            push!(x_coord, s[1])
            push!(y_coord, s[2])
        end
        if i == 1
            scatter!(
                x_coord, y_coord,
                marker=(:circle, 2),
                markercolor=RGB(200 / 255, 200 / 255, 200 / 255),
                markeralpha=0.2,
                label="Particles"
            )
        else 
            scatter!(
                x_coord, y_coord,
                marker=(:circle, 2),
                markercolor=RGB(200 / 255, 200 / 255, 200 / 255),
                markeralpha=0.2,
                label=""
            )
        end
    end
    x_coord = []
    y_coord = []
    push!(x_coord, problem.goal[1])
    push!(y_coord, problem.goal[2])

    scatter!(
        x_coord, y_coord,
        marker=(:circle, 37),
        markeralpha=0.2,
        markercolor=RGB(0 / 255, 255 / 255, 0 / 255),
        markerstrokewidth=0.9,
        label="Goal"
    )

    x_coord = []
    y_coord = []
    push!(x_coord, problem.initialstate.μ[1])
    push!(y_coord, problem.initialstate.μ[2])
    scatter!(
        x_coord, y_coord,
        marker=(:+, 30),
        markercolor=RGB(255 / 255, 204 / 255, 153 / 255),
        markerstrokewidth=0.9,
        label="Start"
    )
    display(p)
    savefig(p, "./plots/belief_traj.pdf")
end

function plot_runtime(experiment_stats::Dict{Int, RuntimeStats})
    categories = categories = string.(collect(keys(experiment_stats)))
    expected_values_pft = [experiment_stats[parse(Int, category)].mean_runtime_no_reuse for category in categories]
    std_dev_pft = [experiment_stats[parse(Int, category)].std_runtime_no_reuse for category in categories]
    expected_values_irpft = [experiment_stats[parse(Int, category)].mean_runtime_reuse for category in categories]
    std_dev_irpft = [experiment_stats[parse(Int, category)].std_runtime_reuse for category in categories]

    # Set up the plot
    bar_width = 0.35
    x = 1:length(categories)
    x1 = x .- bar_width/2
    x2 = x .+ bar_width/2

    # Create the grouped bar plot
    p = bar(x1, expected_values_pft, 
        # yerr=std_dev_algo1,
        bar_width=bar_width,
        label="PFT",
        color=:blue,
        alpha=0.7
    )
    for (xi, yi, ye) in zip(x1, expected_values_pft, std_dev_pft)
        plot!([xi, xi], [yi - ye, yi + ye], lw = 2, color = :black, label = "")  # Line thickness and color
        plot!([xi - 0.1, xi + 0.1], [yi + ye, yi + ye], lw = 2, color = :black, label = "")  # Top cap
        plot!([xi - 0.1, xi + 0.1], [yi - ye, yi - ye], lw = 2, color = :black, label = "")  # Bottom cap
    end

    bar!(x2, expected_values_irpft, 
        # yerr=std_dev_algo2,
        bar_width=bar_width,
        label="IR-PFT",
        color=:red,
        alpha=0.7
    )

    for (xi, yi, ye) in zip(x2, expected_values_irpft, std_dev_irpft)
        plot!([xi, xi], [yi - ye, yi + ye], lw = 2, color = :black, label = "")  # Line thickness and color
        plot!([xi - 0.1, xi + 0.1], [yi + ye, yi + ye], lw = 2, color = :black, label = "")  # Top cap
        plot!([xi - 0.1, xi + 0.1], [yi - ye, yi - ye], lw = 2, color = :black, label = "")  # Bottom cap
    end

    # Customize the plot
    plot!(
        xticks=(x, categories),
        xtickfont = font(20),
        ytickfont = font(20),
        guidefont = font(20),
        legendfont = font(20),
        xlabel="Particles",
        ylabel="Runtime [s]",
        legend=:top
    )
    # Add a bit more space on top of the plot for the error bars
    ylims!(0, maximum([expected_values_pft .+ std_dev_pft; 
                    expected_values_irpft .+ std_dev_irpft]) * 1.1)
    savefig(p, "./plots/runtime_comparison.pdf")
    display(p)
end

function plot_accumulated_reward(experiment_stats::Dict{Int, RuntimeStats})
    categories = categories = string.(collect(keys(experiment_stats)))
    expected_values_pft = [experiment_stats[parse(Int, category)].mean_g_no_reuse for category in categories]
    std_dev_pft = [experiment_stats[parse(Int, category)].std_g_no_reuse for category in categories]
    expected_values_irpft = [experiment_stats[parse(Int, category)].mean_g_reuse for category in categories]
    std_dev_irpft = [experiment_stats[parse(Int, category)].std_g_reuse for category in categories]

    # Set up the plot
    bar_width = 0.35
    x = 1:length(categories)
    x1 = x .- bar_width/2
    x2 = x .+ bar_width/2

    # Create the grouped bar plot
    p = bar(x1, expected_values_pft, 
        yerr=std_dev_pft,
        bar_width=bar_width,
        label="PFT",
        color=:blue,
        alpha=0.7,
    )
    bar!(x2, expected_values_irpft, 
        yerr=std_dev_irpft,
        bar_width=bar_width,
        label="IR-PFT",
        color=:red,
        alpha=0.7
    )
    plot!(
        xticks=(x, categories),
        xlabel="Particles",
        ylabel="Accumulated Reward",
        legend=:topright,
        xtickfont = font(20),
        ytickfont = font(20),
        guidefont = font(20),
        legendfont = font(20),
    )
    savefig(p, "./plots/reward_comparison.pdf")
    display(p)

end

function plot_speedup(experiment_stats::Dict{Int, RuntimeStats})
    categories = categories = collect(keys(experiment_stats))
    expected_values = [experiment_stats[category].mean_speedup for category in categories]
    std_dev = [experiment_stats[category].std_speedup for category in categories]
    p = bar(categories, expected_values,
        yerr=std_dev,
        label="Speedup",
        color=:red,
        alpha=0.7,
        legend=:top,
        xlabel="Particles",
        ylabel="Speedup",
        xtickfont = font(20),
        ytickfont = font(20),
        guidefont = font(20),
        legendfont = font(20),
        ylim=(1.2, 2),
    )
    for (xi, yi, ye) in zip(categories, expected_values, std_dev)
        plot!([xi, xi], [yi - ye, yi + ye], lw = 2, color = :black, label = "")  # Line thickness and color
        plot!([xi - 0.1, xi + 0.1], [yi + ye, yi + ye], lw = 2, color = :black, label = "")  # Top cap
        plot!([xi - 0.1, xi + 0.1], [yi - ye, yi - ye], lw = 2, color = :black, label = "")  # Bottom cap
    end
    savefig(p, "./plots/speedup.pdf")
    display(p)
 
 
 end