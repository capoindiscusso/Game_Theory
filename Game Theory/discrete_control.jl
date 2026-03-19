function exact_greedy_bellman(x, p, G, a, b, c, s, kappa) # greedy bellman with H=1

    return ((s .* x .+ (1.0 .- s) .* (a .- p .+ G * x) ./ b) .- (1.0 .- s) ./ b .* (p .- c)) ./ (2.0 .* ((1.0 .- s) ./ b .+ kappa))

end


function simulate_users!(x, p, G, a, b, beta, s, steps) #Noisy Best Response
    
    N = length(x)
    
    for _ in 1:steps
        i = rand(1:N) # Scegli un utente a caso
        
        # Cambia adozione solo se supera la resistenza s
        if rand() > s[i] 
            mu = (a - p[i] + dot(G[i, :], x)) / b
            sigma = sqrt(1.0 / (beta * b))
            
            # Estrae la nuova adozione dalla Gaussiana
            new_x = rand(Normal(mu, sigma))
            x[i] = max(0.0, new_x) # L'adozione non scende sotto zero
        end
    end
end


function H_step_profit(delta_P_flat, x_init, p_init, G, a, b, c, s, kappa, H, δ)

    #Objective Function: get the total discounted profit for H steps

    N = length(x_init)
    
    #In this secion the actions are the price variations delta_p that at each step has N elements so that in H step has NxH elements
    #Optim optimization works with such action in terms of the "flat vector, here to evaluate the reward we reshape in the matrix form

    delta_P = reshape(delta_P_flat, N, H)
    
    x_curr = copy(x_init)
    p_curr = copy(p_init)
    total_profit = 0.0
    
    for t in 1:H
        dp = delta_P[:, t] # Azioni per il turno t
        
        # Deterministic Best Response
        x_BR = (a .- (p_curr .+ dp) .+ G * x_curr) ./ b
        
        x_next = s .* x_curr .+ (1.0 .- s) .* x_BR
        x_next .= max.(0.0, x_next)
        p_next = p_curr .+ dp
        
        reward = sum((p_next .- c) .* x_next .- kappa .* (dp.^2))
        total_profit += (δ^(t-1)) * reward
        
        x_curr .= x_next
        p_curr .= p_next
    end
    
    #Return negative Cumulative profit as Optim library MINIMIZES by default.
    return -total_profit 
end


function exact_bellman_continuous(x_init, p_init, G, a, b, c, s, kappa, H; δ=0.95) #solve greedy bellman with horizon H with package Optim

    N = length(x_init)
    
    #Start with Hp no price variations for N users in the next H steps
    initial_guess = zeros(N * H)
    
    #We want to minimize the objective function that is the profit as a function of price variations
    objective_function(dp) = H_step_profit(dp, x_init, p_init, G, a, b, c, s, kappa, H, δ)
    
    # Lanciamo l'algoritmo di ottimizzazione non vincolata (L-BFGS è perfetto per funzioni quadratiche)
    result = optimize(objective_function, initial_guess, LBFGS())
    
    # Estraiamo la sequenza ottimale trovata
    optimal_sequence_flat = Optim.minimizer(result)
    optimal_sequence = reshape(optimal_sequence_flat, N, H)
    
    first_optimal_action = optimal_sequence[:, 1]
    
    return first_optimal_action
end


function run_bellman(x_start, p_start, G, a, b, c, beta, s, kappa, T; H = 5)

    N = length(x_start)
    x_hist = zeros(N, T + 1)
    p_hist = zeros(N, T + 1)
    
    x_curr = copy(x_start)
    p_curr = copy(p_start)
    
    x_hist[:, 1] .= x_curr
    p_hist[:, 1] .= p_curr

    cumulative_reward = 0.0
    
    for t in 1:T

        mossa_ottima = exact_bellman_continuous(x_curr, p_curr, G, a, b, c, s, kappa, H)

        p_curr .= max.(c, p_curr .+ mossa_ottima)

        simulate_users!(x_curr, p_curr, G, a, b, beta, s, N * 2)
        
        cumulative_reward += sum((p_curr .- c) .* x_curr .- kappa .* (mossa_ottima.^2))

        x_hist[:, t+1] .= x_curr
        p_hist[:, t+1] .= p_curr
    end
    
    return x_hist, p_hist, cumulative_reward
end

function get_state_idx(x_val::Float64, x_max::Float64, K_s::Int64) # Discretize state space for Q-learning
    bin_width = x_max / K_s
    idx = ceil(Int, x_val / bin_width)
    return clamp(idx, 1, K_s)
end

function train_sarsa!(G, a, b, c, β, s, k, discrete_prices, x_max, K_s, T; episodes=10_000, α=0.1, δ=0.99, ε=0.1)
    
    N = size(G)[1]

    K_a = length(discrete_prices)

    Q = zeros(K_s, K_a)

    for ep in 1:episodes
        
        x = (x_max*rand())*ones(N)
        
        cumulative_reward = 0.0        
        
        state = get_state_idx(mean(x), x_max, K_s)                                #with state we denote here the discretize bin indexes in which the usages x is in
        action = rand() < ε ? rand(1:K_a) : argmax(Q[state, :])         #the action is a vector of discretized prices indexes

        price = discrete_prices[action]*ones(N)

        for t in 1:T  

            simulate_users!(x, price, G, a, b, β, s, N * 2)                                    #update x according to Noisy Best Response
            
            reward = sum(transpose(price).*x)

            next_state = get_state_idx(mean(x), x_max, K_s)
            next_action = rand() < ε ? rand(1:K_a) : argmax(Q[state, :])


            Q_old = Q[state, action]
            Q_next = Q[next_state, next_action]
                
            Q[state, action] = Q_old + α*(reward + δ*Q_next - Q_old)
                
            cumulative_reward += sum(reward)
            
            state = next_state
            action = next_action
        end
        
        if rem(ep,1_000)==0 || ep== 1
            ε *= 0.95
            println("Episode $ep    Cumulative reward: $(round(cumulative_reward, digits=2))        ε: $(round(ε, digits=3))")
        end

    end

    return Q
end

function simulateSarsa(Q, x, G, a, b, c, β, s, k, discrete_prices, x_max, K_s, T)


    N = length(x)
    K_a = length(discrete_prices)

    X = zeros(N,T)
    P = zeros(N,T)
        
    cumulative_reward = 0.0  
    k = 0.0      
        
    state = get_state_idx(mean(x), x_max, K_s)
    action = argmax(Q[state, :])
    price = discrete_prices[action]*ones(N)

    X[:,1] = x
    P[:,1] = price
   

    for t in 2:T

        simulate_users!(x, price, G, a, b, β, s, N * 2)  #update x according to Noisy Best Response
        
        state = get_state_idx(mean(x), x_max, K_s)
        action = argmax(Q[state, :])

        price = discrete_prices[action]*ones(N)

        X[:,t] = x
        P[:,t] = price

        cumulative_reward += transpose(price .- c)*x

    end
    
    println("Episode generated successfully following Q valuie function from SARSA training.\nTotal Reward: $(round(cumulative_reward; digits=2))")
    return  X, P
end