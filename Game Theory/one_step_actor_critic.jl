using LinearAlgebra
using Distributions

function monopolist_action(state, μ, σ)
    shift = rand(Normal(0.0, σ))
    price = exp(μ + shift) * ones(N) 
    return price, shift
end

function monopolist_euristic(state, price)

    price += 0.5(p_star - price)

end

function evolvi_stato(state, price; noisy=false)

    new_state = state + Λ*((a*ones(N) - price + G*state)/b - state)
    
    if noisy
        for i in 1:1
            j = rand(1:N)
            new_state[j] = max(0.0, new_state[j] + rand(Normal(0.0, 0.1*σ)))
        end
    end

    new_state
end

function get_reward(state, price)
    (transpose(price - c*ones(N)) * state)
end

function generate_episode(θ, state_0, T; euristic=false)

    states = zeros(N,T)
    prices = zeros(N,T)
    rewards = zeros(T)
    

    states[:,1] = state_0

    x = feature_state(states[:,1])
    μ = transpose(θ)*x

    prices[:,1] = monopolist_action(state_0, μ, σ)[1]


    for t in 2:T

        states[:,t] = evolvi_stato(states[:,t-1], prices[:,t-1]; noisy=false)

        x = feature_state(states[:,t])
        μ = transpose(θ)*x

        prices[:,t] = monopolist_action(states[:,t], μ, σ)[1]

        if euristic
            prices[:,t] = monopolist_euristic(states[:,t], prices[:,t-1])
        end

        rewards[t] = get_reward(states[:,t-1], prices[:,t-1])
    end

    total_reward = sum(rewards)

    println("Episode generated successfully. Total Reward: $(round(total_reward; digits=2))")

    return states, prices, rewards

end

function feature_state(state)
    vcat(1,[state[i]*state[j] for i in 1:N for j in i:N])
end

function parametric_v(state, w)
    transpose(w)*feature_state(state)
end

function one_step_actor_critic(α_θ, α_w, γ, σ, T, num_training)

    d_prime = length(feature_state(state_star))

    θ = 0.00001*ones(d_prime)
    w = zeros(d_prime)
    rumoroso = false

    #Let's suppose knows at least N and parameters a,b,c
    #x_target = a/(4*(N-1-b/2))

    #println("x_target: $x_target")

    for i in 1:num_training
        
        #state = rand(Normal(x_target, 0.1), N)
        state = (1/N)*(ones(N) + 0.1*rand(N) - 0.1*rand(N))

        I = 1.0

        if i > 5000
            rumoroso = true
        end

        for t in 1:T
            x = feature_state(state)
            μ = transpose(θ)*x
            
            # Ottieni il prezzo e il rumore specifico (ε)
            price, shift = monopolist_action(state, μ, σ)

            reward = get_reward(state, price) 
            
            new_state = evolvi_stato(state, price; noisy=rumoroso)

            δ = clamp(reward + γ*parametric_v(new_state, w) - parametric_v(state, w), -10.0, 10.0)
            
            # Aggiornamento Critic
            w += clamp.(α_w * δ * x, -1.0, 1.0) 

            # Aggiornamento Actor (Score function per Log-Normal)
            # Il gradiente di log(π) rispetto a μ quando p = exp(μ + ε) è (shift / σ^2)
            θ += α_θ * I * δ * (shift / σ^2) * x

            I *= γ
            state = new_state
        end

        if rem(i,1000)==0 || (rem(i,100)==0 && i<= 1000) || i==1

            mu = transpose(θ)*feature_state(state)

            v = transpose(w)*feature_state(state)

            println("Episode $i exp(mu): $(round(exp(mu); digits=3)) v: $(round(v; digits=3))")

        end

    
    end

    return θ
end