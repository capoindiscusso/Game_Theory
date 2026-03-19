using Plots

function plot_graph(G; titolo="", lay=:stress)

    graphplot(G, 
        names=1:size(G,1), 
        nodesize=0.3,
        method = lay, 
        curves=false, 
        markercolor = :lightgrey, 
        title = titolo,
        fontsize = 8)
end

function plot2graphs(G, p_star, x_star; lay = :circular)
    
    x_min, x_max = 0.9*minimum(x_star), maximum(x_star)*1.1
    p_min, p_max = 0.9*minimum(p_star), 1.1*maximum(p_star)



    p1 = graphplot(G, 
        names = 1:N, 
        nodesize = 0.25, 
        curves = false, 
        method = lay,      # FORZA il layout
        marker_z = p_star,          #gradazione in base a p_star
        markercolor = :viridis,     
        clims = (p_min, p_max),      
        colorbar = true,             
        title = "Optimal Prices",
        fontsize = 8
    )

    p2 = graphplot(G, 
        names = 1:N, 
        nodesize = 0.25, 
        curves = false, 
        method = lay,      # FORZA lo stesso layout
        marker_z = x_star,     #gradazione in base a x_star
        markercolor = :plasma,     
        clims = (x_min, x_max),      
        colorbar = true,             
        title = "Consumers' usage at NE",
        fontsize = 8
    )

    # Unione dei due plot
    plot_finale = plot(p1, p2, 
        layout = Plots.grid(1, 2), 
        size = (800, 400),
        margin = 10Plots.mm          # Margine aumentato per la legenda
    )
    display(plot_finale)
end

function plot_discrete_episode(states, prices, T_0, T_F; p_min = 0.0, p_max = 10.0 )

    plot_states = plot()

    for i in 1:N
        plot!(states[i,:], label="consume_$i")
    end

    title!("Consumers' usage")
    xlims!(T_0, T_F)
    xlabel!("t")
    ylabel!("x")

    plot_prices = plot()

    for i in 1:N
        plot!(prices[i,:], label="price_$i")
    end

    title!("Prices")
    xlims!(T_0, T_F)
    ylims!(p_min, p_max)
    xlabel!("t")
    ylabel!("p")

    plot(plot_states, plot_prices, layout=(2,1), legend=false)

end

function plot_continuous_episode(states, prices, rewards, times, T_0, T_F)

    plot_states = plot()
    plot!(times, states[1,:], label="consume")
    plot!(times, x_star[1,:], label="x_star", linestyle=:dash)
    
    for i in 2:N
        plot!(times, states[i,:], label="")
        plot!(times, x_star[i,:], label="", linestyle=:dash)
    end

    title!("Consumers' usage")
    xlims!(T_0, T_F)
    xlabel!("t")
    ylabel!("x")

    plot_prices = plot()

    for i in 1:N
        plot!(times, prices[i,:], label="price_$i")
    end

    title!("Prices")
    xlims!(T_0, T_F)
    xlabel!("t")
    ylabel!("p")

    p_x_p = plot(plot_states, plot_prices, layout=(2,1), legend=true, size=(800,800))

    p_r = plot(times, rewards, label="cumulative reward")
    title!("Rewards")
    xlims!(T_0, T_F)
    xlabel!("t")
    ylabel!("r")

    l = @layout [a{0.5w} b]
    plot_final = plot(p_x_p, p_r, layout = l, size=(1200, 800))
    display(plot_final)
end

