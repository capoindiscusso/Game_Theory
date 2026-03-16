using Plots

function plot_risultati(states, prices, T_0, T_F)

    plot_states = plot()

    for i in 1:N
        plot!(states[i,:], label="consume_$i")
    end

    title!("Consumers' usage")
    xlims!(1, T)
    xlabel!("t")
    ylabel!("x")

    plot_prices = plot()

    for i in 1:N
        plot!(prices[i,:], label="price_$i")
    end

    title!("Prices")
    xlims!(1, T)
    xlabel!("t")
    ylabel!("p")

    plot(plot_states, plot_prices, layout=(2,1), legend=false)

end