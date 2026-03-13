function exact_greedy_bellman(x, p, G, a, b, c, s, kappa) # greedy bellman with H=1
    return ((s .* x .+ (1.0 .- s) .* (a .- p .+ G * x) ./ b) .- (1.0 .- s) ./ b .* (p .- c)) ./ (2.0 .* ((1.0 .- s) ./ b .+ kappa))
end


function simulate_users!(x, p, G, a, b, beta, s, steps) #NBR
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

# 1. Funzione Obiettivo: calcola il profitto totale scontato per H passi
function H_step_profit(delta_P_flat, x_init, p_init, G, a, b, c, s, kappa, H, δ)
    N = length(x_init)
    
    # L'ottimizzatore lavora con vettori "piatti". Lo rimodelliamo in una matrice [N utenti x H passi]
    delta_P = reshape(delta_P_flat, N, H) 
    
    x_curr = copy(x_init)
    p_curr = copy(p_init)
    total_profit = 0.0
    
    for t in 1:H
        dp = delta_P[:, t] # Azioni per il turno t
        
        # Dinamica Deterministica
        x_NE = (a .- (p_curr .+ dp) .+ G * x_curr) ./ b
        
        x_next = s .* x_curr .+ (1.0 .- s) .* x_NE
        x_next .= max.(0.0, x_next) # Le adozioni non possono essere negative
        p_next = p_curr .+ dp
        
        # Calcolo del profitto
        reward = sum((p_next .- c) .* x_next .- kappa .* (dp.^2))
        total_profit += (δ^(t-1)) * reward
        
        # Avanzamento dello stato per il turno successivo
        x_curr .= x_next
        p_curr .= p_next
    end
    
    # Restituiamo il profitto col segno meno, perché la libreria Optim MINIMIZZA di default.
    # Minimizzare -Profitto equivale a Massimizzare il Profitto.
    return -total_profit 
end


function exact_bellman_continuous(x_init, p_init, G, a, b, c, s, kappa, H; δ=0.95) #solve greedy bellman with horizon H with package Optim
    N = length(x_init)
    
    # Initial guess: partiamo con l'ipotesi di non cambiare nessun prezzo (tutti zeri)
    # Dimensioni: N utenti * H passi di tempo
    initial_guess = zeros(N * H)
    
    # Creiamo una funzione anonima (closure) che dipenda SOLO da delta_P_flat
    # (è il formato che richiede la libreria Optim)
    objective_function(dp) = H_step_profit(dp, x_init, p_init, G, a, b, c, s, kappa, H, δ)
    
    # Lanciamo l'algoritmo di ottimizzazione non vincolata (L-BFGS è perfetto per funzioni quadratiche)
    result = optimize(objective_function, initial_guess, LBFGS())
    
    # Estraiamo la sequenza ottimale trovata
    optimal_sequence_flat = Optim.minimizer(result)
    optimal_sequence = reshape(optimal_sequence_flat, N, H)
    
    max_profit = -Optim.minimum(result)
    first_optimal_action = optimal_sequence[:, 1]
    
    return max_profit, first_optimal_action
end


function run_simulation(x_start, p_start, G, a, b, c, beta, s, kappa, H, T_sim)
    N = length(x_start)
    x_hist = zeros(N, T_sim + 1)
    p_hist = zeros(N, T_sim + 1)
    
    x_curr = copy(x_start)
    p_curr = copy(p_start)
    
    x_hist[:, 1] .= x_curr
    p_hist[:, 1] .= p_curr
    
    for t in 1:T_sim
        # Calcolo mossa ottima continua
        _, mossa_ottima = exact_bellman_continuous(
            x_curr, p_curr, G, a, b, c, s, kappa, H
        )
        
        # Applichiamo i nuovi prezzi
        p_curr .= max.(c, p_curr .+ mossa_ottima)
        
        # Reazione della rete
        simulate_users!(x_curr, p_curr, G, a, b, beta, s, N * 2)
        
        # Salvataggio dati
        x_hist[:, t+1] .= x_curr
        p_hist[:, t+1] .= p_curr
    end
    
    return x_hist, p_hist
end

function get_state_idx(x_val, x_max, K_s) # Discretize state space for Q-learning
    bin_width = x_max / K_s
    idx = ceil(Int, x_val / bin_width)
    return clamp(idx, 1, K_s)
end

function train_sarsa!(x, p, G, a, b, c, beta, s, kappa, actions, x_max, K_s; 
                      episodes=500, steps_per_episode=50, alpha=0.1, δ=0.95, epsilon=0.1)
    
    N = length(x)
    K_a = length(actions)
    
    Q = zeros(N, K_s, K_a)
    
    total_steps = episodes * steps_per_episode
    x_history = zeros(N, total_steps + 1)
    p_history = zeros(N, total_steps + 1)
    
    x_history[:, 1] .= x
    p_history[:, 1] .= p
    time_idx = 2
    
    for ep in 1:episodes
        p_current = copy(p) 
        
        S = [get_state_idx(x[i], x_max, K_s) for i in 1:N]
        A = [rand() < epsilon ? rand(1:K_a) : argmax(Q[i, S[i], :]) for i in 1:N]
        
        for t in 1:steps_per_episode
            # Applichiamo i prezzi
            delta_p = [actions[A[i]] for i in 1:N]
            p_current .= max.(c, p_current .+ delta_p)
            
            # Reazione degli utenti
            simulate_users!(x, p_current, G, a, b, beta, s, N * 2)
            
            # Salviamo la storia
            x_history[:, time_idx] .= x
            p_history[:, time_idx] .= p_current
            time_idx += 1
            
            # Osserviamo il nuovo stato e scegliamo la nuova azione
            S_next = [get_state_idx(x[i], x_max, K_s) for i in 1:N]
            A_next = [rand() < epsilon ? rand(1:K_a) : argmax(Q[i, S_next[i], :]) for i in 1:N]
            
            # Aggiornamento Q-Table (SARSA)
            for i in 1:N
                reward = (p_current[i] - c) * x[i] - kappa * (delta_p[i]^2)
                
                Q_old = Q[i, S[i], A[i]]
                Q_next = Q[i, S_next[i], A_next[i]]
                
                Q[i, S[i], A[i]] = Q_old + alpha * (reward + δ * Q_next - Q_old)
            end
            
            S .= S_next
            A .= A_next
        end
    end
    
    return Q, x_history, p_history
end