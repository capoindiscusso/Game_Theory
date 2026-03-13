# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

# **Author:** Agenti Interagenti  
# **Date:** March 2026  
#
# # <center>Optimal Pricing on Network</center>

#
#
# ## <center>Introduction</center>
#
# A monopolist sells a divisible good to $N$ consumers. The consumers are connected via a network $G=(V,E)$ defined by an adjacency matrix $G=(g_{ij})$, $i,j=1,\dots,N$. Each consumer $i$’s usage level $x_i$ depends on the goods of her neighbors through a positive externality.  
# Consumer $i$’s utility is linear-quadratic:
#
# $$u_i(x_i,x_{-i}) = ax_i - \frac{b}{2}x_i^2 + \sum_j g_{ij}x_ix_j - p_ix_i$$
#
# where $x_i$ is the usage of consumer $i$, $g_{ij}$ is the weight of influence from $j$ to $i$, $p_i$ is the price offered to $i$. Also $a>0$ is the intrinsic utility parameter, and $b>0$ is the sensitivity to own consumption. Consumers simultaneously choose usages to maximize their utilities given the price vector $p=(p_1,\dots,p_N)$. The monopolist sets $p$ in order to maximize the revenue:
#
# $$R(p) = \sum_i (p_i - c)x_i$$
#
# where $c \geq 0$ is the marginal cost of producing the good. We assume $a>c$ to ensure positive demand in isolation, and impose $b > \sum_j g_{ij} \ \forall i$ to guarantee a unique, stable equilibrium.
#
# ### 1. Consumer Best Response
#
# Consumer's utility function is $u_i(x_i,x_{-i}) = ax_i - \frac{b}{2}x_i^2 + \sum_j g_{ij}x_ix_j - p_ix_i$, that is concave in $x_i$, so to find the Nash equilibrium it's sufficient the first-order condition (assuming that the action space of each consumer is compact, and this is reasonable because it's bounded by $0$ and the total amount of the produced good):
#
# $$\begin{aligned}
# \frac{\partial u_i}{\partial x_i} &= (a - p_i) - b x_i + \sum_j g_{ij} x_j = 0 \\
# x_i^* &= \frac{a - p_i}{b} + \frac{1}{b} \sum_j g_{ij} x_j
# \end{aligned}$$
#
# In vectorial form this is:
#
# $$\begin{aligned}
# x^* &= \frac{a𝟙 - p}{b} + \frac{1}{b} G x^* \\
# x^* &= \left(I - \frac{1}{b}G\right)^{-1} \frac{a𝟙 - p}{b}
# \end{aligned}$$
#
# where $𝟙$ is a $N$-dimensional vector of all $1$. 
# Raga per me questo align sotto qui si può anche togliere non l'ho capito molto
# $$\begin{aligned}
#     \left(I - \frac{1}{b}G\right)x &= \frac{a𝟙 - p}{b} \\
#     y &= \frac{a𝟙 - p}{b} \implies \left(I - \frac{1}{b}G\right)x = y \\
#     \tilde{x}_1 &= \frac{\tilde{y}_1}{1 - \frac{\lambda_1}{b}} \\
#     x^* &= \tilde{x}_1 v_1 + \sum_{k \geq 2} \tilde{x}_k v_k
# \end{aligned}$$
#
# To have an unique and stable equilibrium, $I - \frac{1}{b}G$ has to be invertible. This can be proved defining the best-response operator
#
# $$T(x) = \frac{a𝟙 - p}{b} + \frac{1}{b} G x$$
#
# that gives the best response given the action vector $x$. If this operator is contractive, the best-response dynamics converges to an unique, stable equilibrium. Using the norm
#
# $$\begin{aligned}
# ||x||_\infty &\doteq \max_i |x_i| \\
# ||A||_\infty &\doteq \max_i \sum_j |A_{ij}|
# \end{aligned}$$
#
# the operator $T(x)$ is contractive if $\exists k \in [0,1)$ such that $||T(x) - T(y)||_\infty \leq k||x-y||_\infty$, $\forall x,y$, and this is true if
#
# $$ \|T(x) - T(y)\|_\infty = \| \frac{1}{b}G(x-y) \|_\infty \leq \frac{1}{b} \|G\|_\infty \cdot \|x-y\|_\infty $$
#
# where we used the property $||Ax||_\infty \leq ||A||_\infty \cdot ||x||_\infty$. Identifying $\frac{1}{b} ||G||_\infty$ with $k$, $0 \leq k < 1$ is equivalent to the condition
#
# $$b > \sum_j g_{ij} \quad \forall i$$
#
# For Perron-Frobenius theorem, we have that $G$ is invertible because its elements are non-negative, and its spectral radius satisfies
#
# $$\min_i \sum_j g_{ij} \leq \rho(G) \leq \max_i \sum_j g_{ij}$$
#
# So, if $b > \sum_j g_{ij} \quad \forall i$, we have that $\rho(G) < 1$ and $(I - \frac{1}{b}G)^{-1} = \sum_{i=0}^\infty \left(\frac{1}{b}G\right)^i$.
#
# ### 2. Optimal Pricing
#
# If agents consume according to the Nash equilibrium, defining $M = (I - \frac{1}{b}G)^{-1}$, the revenue for the monopolist is
#
# $$\begin{aligned}
# R(p) &= (p - c𝟙)^\top x^* = (p - c𝟙)^\top M \frac{a𝟙 - p}{b}\\
# &= \frac{1}{b} \left[ -p^\top M p + p^\top M 𝟙 a + 𝟙^\top M p c - 𝟙^\top M 𝟙 ac \right]
# \end{aligned}$$
#
# To maximize $R(p)$ it's sufficient to impose $\nabla_p R = 0$
#
# $$\begin{aligned}
# \nabla_p R &= -(M + M^\top)p^* + M𝟙a + M^\top𝟙c = 0 \\
# \Rightarrow p^* &= (M + M^\top)^{-1} [aM + cM^\top]𝟙 \\
# &= c𝟙 + (a-c)(M + M^\top)^{-1} M 𝟙
# \end{aligned}$$
#
# From the definition of $M$, if $b > \sum_j g_{ij} \ \forall i$
#
# $$M = \left(I - \frac{1}{b}G\right)^{-1} = \sum_{i=0}^\infty \left(\frac{1}{b}G\right)^i = I + \frac{1}{b}G + \frac{1}{b^2}G^2 + \dots$$
#
# The element $M_{ij}$ represents the total influence (both direct and indirect) that agent $j$ exerts on agent $i$. In this expansion, the term $(G^k)_{ij}$ counts, with unitary link wieghts, the number of paths of length $k$ between node $j$ and node $i$. The scalar $(1/b)^k$ discounts the impact of longer connections and so $M_{ij}$ represents the global influence of $j$ on $i$.
#
# Interestingly, we notice that when agents are linked through an undirected network, i.e. when $G$ is symmetric, $M$ is symmetric as well, namely $M = M^\top$, so that the global influence of $j$ on $i$ is equal to the global influence of $i$ on $j$. This implies that in such cases the expression for the optimal price vector is extremely simplified
#
# $$\begin{aligned}
# p^* &= c𝟙 + (M + M^\top)^{-1} M 𝟙 (a-c) \\
# &= c𝟙 + (2M )^{-1} M 𝟙 (a-c) \\
# &=  c𝟙  + \frac{1}{2} 𝟙 (a-c) \\
# &= \frac{1}{2}(a+c) 𝟙
# \end{aligned}$$
#
# as it does not depend on the network topology and all its elements have the same value, that corresponds to the monopolist fixing the same optimal price for each agent. We can than interpret it as a sort of basis or unbiased price. We shall see however that, fixed such a price vector, this does not imply that the usage of each agent in the Nash equilibrium is the same, as it still depends on the network.
#
# Furthermore, by simply looking at the general expression of $p^* = c𝟙 + (M + M^\top)^{-1} M 𝟙 (a-c)$, we can establish a link between the optimal price vector and the Katz centrality. We recall that, given the adjacency matrix $G$, $\lambda_G$ being its larger eigenvalue, Katz centrality measures the "influence" or "importance" of each node and it's defined as
#
# $$\begin{aligned}
# z &= (1-\beta) \lambda_G^{-1} G z + \beta \mu \\
# z &= \left( I - (1-\beta) \lambda_G^{-1} G \right)^{-1} \beta \mu
# \end{aligned}$$
#
# where $\beta \in [0,1]$ and $\mu \in \mathbb{R}^N$ is the intrinsic centrality of each node. In our problem this is very similar to the term $M𝟙(a-c)$, so in the expression of the optimal price we can interprete the term $c𝟙$ as a basis price for everyone, plus the term $(M+M^\top)^{-1}M𝟙(a-c)$, that is proportional to the difference between the intrinsic utility parameter $a$ and the marginal cost $c$ and where $M𝟙$ measures how much the agents are influenced by other agents in the network. We can conclude that the agents that are more influenced by others will get higher prices, while agents that have a strong influence will have lower prices. We'll deepen this consideration studying the problem on different networks in the next section.
#
# We can rewrite the best response vector $x^*$ assuming that the monopolist applies optimal prices, i.e. in an overall equilibrium condition
#
# $$\begin{aligned}
# x^* &= M \frac{a𝟙 - p^*}{b} \\
# &=  M 𝟙 \frac{a-c}{b} - M (M + M^\top)^{-1} M 𝟙 \frac{a-c}{b} \\
# &= \frac{a-c}{b}\left( I - M (M + M^\top)^{-1} \right)  M 𝟙
# \end{aligned}$$
#
# If the graph is undirected
#
# $$x^* = M \frac{a𝟙 - p^*}{b} = M\frac{a𝟙 - \frac{1}{2}(a+c)𝟙}{b} = \frac{1}{2}\frac{a-c}{b} M 𝟙$$
#
# we see that, as mentioned before, the consumers' best response still depends on $M = (I - \frac{1}{b}G)^{-1}$, namely on the network topology.

# +
using Graphs, GraphRecipes
using Plots
using SparseArrays, LinearAlgebra

include("myfunctions.jl")
  
"""PARAMETERS"""
N = 9
a = 10.0
b = 12.0
c = 2.0 

"""GRAPH INITIALIZATION"""

#G = createGraphLattice(N,N; periodic=false)
#G = createGraphStar(N)
#G = createGraphErdos(N, sparse=false)
G = createGraphFullyConnected(N)

checkParameters(G, a, b, c)

include("myfunctions.jl")

#RANDOM USAGE AND PRICE INITIALIZATION
my_p = 11.0*ones(N)
my_x = 1.1*ones(N) + 0.1*rand(N)

my_utility = utility(my_x, my_p, G, a, b)
println("Consumers utilities are ", my_utility)

R = reward(my_x, my_p, c)
println("Reward is ", R)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Nash EQ found to be ", round.(x_star, digits = 4))

p_star = bestPrice(M, a, b, c)
println("Optimal Price is ", round.(p_star, digits = 4))

R_star = reward(x_star, p_star, c)

graphplot(G, names=1:size(G,1), nodesize=0.3, curves=false, markercolor = :lightgrey)

graphplot(G, names=1:size(G,1), nodesize=0.3, edgelabel=round.(G,digits=2), markercolor = :lightgrey)

graphplot(G, names=1:size(G,1), nodesize=0.3, curves=false, method=:shell, markercolor = :lightgrey)

graphplot(G, names=1:size(G,1), nodesize=0.3, method=:shell, edgelabel=round.(G,digits=2), markercolor = :lightgrey)

# +
using LinearAlgebra, Plots, Graphs

N = 15
a, b, c = 20.0, 12.0, 1.0  
α, η = 0.2, 0.05           # Reattività utenti e monopolista
T = 300

# GRAFO
 g = erdos_renyi(N, 0.25)
# g = star_graph(N) 
G = Float64.(adjacency_matrix(g))

for i in 1:N, j in 1:N
    if G[i,j] > 0 G[i,j] = 2.5 end # Peso uniforme 2.5
end
 
x = fill(0.5, N)
p = fill(a/2, N)
history_p = zeros(T, N)
out_influence = sum(G, dims=1)[:] 

for t in 1:T

    x = (1 - α) .* x .+ α .* max.(0.1, (a .- p .+ G * x) ./ b)
    
    grad = (x .- (p .- c) ./ b) .- 1.0 .* out_influence .* (p .- c) ./ b
    p .+= η .* grad
    p .= max.(c + 0.1, p)
    
    history_p[t, :] = p
end

function get_coords_with_arcs(g)
    n = nv(g); pos = rand(2, n) .* 2.0
    for _ in 1:200 
        for i in 1:n
            f = zeros(2)
            for j in 1:n
                if i == j continue end
                d = pos[:, i] - pos[:, j]; dist = norm(d) + 0.01
                f += (d / (dist^4)) * 0.02 # Repulsione forte per distanziare
                if has_edge(g, i, j) f -= d * dist * 0.4 end # Attrazione elastica
            end
            pos[:, i] += clamp.(f, -0.05, 0.05)
        end
    end
    return pos[1, :], pos[2, :]
end

nx, ny = get_coords_with_arcs(g)

# Plotting

# Plot 1: Traiettorie dei Prezzi (Tutte le linee visibili e con legenda)
p1 = plot(history_p, title="Evoluzione Prezzi su Grafo Erdős-Rényi", lw=2, 
          palette=:tab20, legend=:outerright, 
          labels=reshape(["Nodo $i" for i in 1:N], 1, N))

# Plot 2: Il Grafo con Archi e Labels
p2 = plot(title="Topologia della Rete e Mappa dei Prezzi (Viola=Economico)", 
          axis=false, grid=false, aspect_ratio=:equal, legend=false)

# DISEGNO ARCHI: Grigi, Visibili (lw=1.3)
for e in edges(g)
    u, v = src(e), dst(e)
    plot!(p2, [nx[u], nx[v]], [ny[u], ny[v]], color=:gray, lw=1.3, alpha=0.35)
end

# DISEGNO NODI: Dimensione fissa, Colore = Prezzo
scatter!(p2, nx, ny, marker_z = p, markercolor = :viridis, 
         markersize = 12, colorbar = true, markerstrokewidth=0)

# AGGIUNTA LABELS IDENTIFICATIVE
price_labels = [string("N", i, ":\n", round(p[i], digits=1)) for i in 1:N]
annotate!(p2, [(nx[i], ny[i] + 0.18, text(price_labels[i], 7, :black, :center, :bold)) for i in 1:N])

plot(p1, p2, layout=(2,1), size=(850, 1100), margin=10Plots.mm)

# +
using Random, Distributions, LinearAlgebra
using Graphs, GraphRecipes
using Plots
using SparseArrays

include("myfunctions.jl")

# Parametri generali
N = 9
a = 15.0
b = 20.0
c = 3.0

α = 1e-5           # learning rate
γ = 1.0            # discount factor (per reward)
ξ = 0.9            # discount factor (per learning rate)
n_episodes = 100
T = 300

Λ = 1.0*diagm(ones(N)) #.+ 0.5*rand(N)
#g = createGraphStar(N)
g = createGraphFullyConnected(N)/N
#g = createGraphStarNoLeavesLinks(N)  # ma sta funzione esiste ????
checkParameters(g, a, b, c)


# Parametri della policy
# θ = [θ_μ0, θ_μ1, θ_μ2, θ_log(σ0), θ_log(σ1), θ_log(σ2)]
# inizializzare μ = 0 e log(σ) piccoli e negativi
θ = hcat(zeros(N,4), fill(-3.0,N,4))  # N x 8


# Funzione reward
reward(x,p) = sum((p[i] - c)*x[i] for i in 1:N)

# Simulazione episodio
function run_episode(θ)
    x = collect(range(0,2.0,length=N))
    states = Vector{Vector{Float64}}()
    actions = Vector{Vector{Float64}}()
    rewards = Float64[]

    for t in 1:T
        μ = zeros(N)
        σ = zeros(N)
        p = zeros(N)

        for k in 1:N
            μ[k] = θ[k,1] + θ[k,2]*x[k] + θ[k,3]*x[k]^2 + θ[k,4]*sum(g[k, j] for j in 1:N)
            σ[k] = exp(θ[k,5] + θ[k,6]*x[k] + θ[k,7]*x[k]^2 + θ[k,8]*sum(g[k, j] for j in 1:N))  # parametrizzazione esponenziale garantisce σ>0 sempre
            σ[k] = max(σ[k], 1e-4)                              # clipping per sicurezza
            p[k] = rand(LogNormal(μ[k], σ[k]))
        end

        push!(states, copy(x))
        push!(actions, copy(p))
        push!(rewards, reward(x,p))

        # prossimo stato
        for i in 1:N
            x[i] = x[i] + Λ[i,i]*((a-p[i])/b + sum(g[i,j]*x[j] for j in 1:N)/b - x[i])
            #x[i] = max(x[i], 1e-3)
        end
    end

    return states, actions, rewards
end

# Gradiente log-policy
function grad_log_pi(x,p,θ)
    grad = zeros(N,8)

    for k in 1:N
        μ_k = θ[k,1] + θ[k,2]*x[k] + θ[k,3]*x[k]^2 + θ[k,4]*sum(g[k, j] for j in 1:N)
        logσ_k = θ[k,5] + θ[k,6]*x[k] + θ[k,7]*x[k]^2 + θ[k,8]*sum(g[k, j] for j in 1:N)
        σ_k = exp(logσ_k)

        # gradiente rispetto a parametri di μ
        coeff_mu = (log(p[k]) - μ_k) / σ_k^2
        grad[k,1] = coeff_mu
        grad[k,2] = coeff_mu * x[k]
        grad[k,3] = coeff_mu * x[k]^2
        grad[k,4] = coeff_mu * sum(g[k, :])

        # gradiente rispetto a parametri di log(σ)
        coeff_logσ = ((log(p[k]) - μ_k)^2 / σ_k^2) - 1
        grad[k,5] = coeff_logσ
        grad[k,6] = coeff_logσ * x[k]
        grad[k,7] = coeff_logσ * x[k]^2
        grad[k,8] = coeff_logσ * sum(g[k, :])
    end

    # Clipping
    grad[:,1:4] = clamp.(grad[:,1:4], -0.2, 0.2)  # μ
    grad[:,5:8] = clamp.(grad[:,5:8], -0.05, 0.05) # logσ

    return grad
end

# REINFORCE
for episode in 1:n_episodes
    states, actions, rewards = run_episode(θ)

    # Calcola return cumulativo G_t
    G = zeros(T)
    for t in 1:T
        G[t] = sum((γ .^ (0:(T-t))) .* rewards[t:T])    # Reward cumulativo da ogni stato dell'episodio in poi
        G[t] = G[t] / maximum(abs.(G[t:T]))   # scala [-1,1]
    end

    # Aggiorna θ passo passo con clipping
    for t in 1:T
        grad = grad_log_pi(states[t], actions[t], θ)
        #baseline = mean(G)
        θ += α * (G[t]) * grad
    end
end

println("θ finale dopo training:\n", θ)


# Generazione singolo episodio
x = collect(range(0,5,length=N))   # stato iniziale randomico
T_plot = 200                        # lunghezza episodio per il plot
actions_history = zeros(N, T_plot)  # prezzi ad ogni passo
states_history = zeros(N, T_plot)  # stati ad ogni passo

for t in 1:T_plot
    μ = zeros(N)
    σ = zeros(N)
    p = zeros(N)
    for k in 1:N
        μ[k] = θ[k,1] + θ[k,2]*x[k] + θ[k,3]*x[k]^2
        σ[k] = exp(θ[k,4] + θ[k,5]*x[k] + θ[k,6]*x[k]^2)
        p[k] = rand(LogNormal(μ[k], σ[k]))
    end
    actions_history[:, t] = p   # salva le azioni
    states_history[:, t] = x    # salva gli stati

    # Update stato
    for i in 1:N
        x[i] = x[i] + Λ[i,i]*((a-p[i])/b + sum(g[i,j]*x[j] for j in 1:N)/b - x[i])
        x[i] = x[i] = max(x[i], 1e-3)
    end
end

# Plot dei prezzi per ciascun giocatore

plot()
for i in 1:N
    plot!(1:T_plot, actions_history[i, :], label="Giocatore $i", lw=2)
end
xlabel!("Time step")
ylabel!("Prezzo")
title!("Evoluzione del prezzo per ciascun giocatore")
# -

# ##  <center>Analytical Results and Simulations on Different Networks</center>
#
# ### Fully Connected Graph
#
#
#
# In a *fully connected graph* each node is linked through undirected links to all its $N-1$ neighbors, namely $G_{ij} = (1-\delta_{ij}) \ \forall i,j = 1\dots N$. Defining $J = 𝟙 𝟙^\top$ the $N \times N$ matrix whose elements are all $1$, we have that $G = J-I$. 
#
# Consequently $M = \left(I - \frac{1}{b}G\right)^{-1}$ can be rewritten as:
#
# $$M = \left(I+\frac{(I-J)}{b}\right)^{-1} = \left(\frac{b+1}{b}I-\frac{1}{b}J\right)^{-1}$$
#
# Looking for $M$ of the form $M = xI + yJ$, imposing the defining condition for the inverse matrix:
#
# $$(xI + yJ)\left(\frac{b+1}{b}I -\frac{1}{b}J\right) = I$$
#
# and exploiting the property $J^2 = 𝟙(𝟙^\top𝟙)𝟙^\top = N𝟙𝟙^\top = NJ$ we get:
#
# $$\begin{aligned}
#     x &= \frac{b}{b+1} \\
#     y &= \frac{b}{(b+1)(b+1-N)}
# \end{aligned}$$
#
# Hence we obtain that the matrix $M = \left(I - \frac{1}{b}G\right)^{-1}$ for a fully connected graph takes the form:
#
# $$M =\frac{b}{b+1}I +\frac{b}{(b+1)(b+1-N)}J$$
#
# Having found an explicit expression for $M$, we can now study how the consumers' usages and the optimal price vector are influenced by the network. 
# In this case the optimal price vector, as for any other undirected network, takes the same form $p^* = \frac{1}{2}(a+c) 𝟙$, in which actually the network does not play any role.
#
# Instead, the consumers' usage Nash Equilibrium corresponding to the optimal price vector is:
#
# $$x^* =  \frac{1}{2}\frac{a-c}{b} M 𝟙 = \frac{1}{2}\frac{(a-c)}{b}\left(\frac{b}{b+1}I +\frac{b}{(b+1)(b+1-N)}J\right) 𝟙$$
#
# Noticing that $J𝟙 =N𝟙$, we get:
#
# $$x^* = \frac{1}{2}\frac{(a-c)}{b+1}\left(1 +\frac{N}{b+1-N}\right) 𝟙 = \frac{1}{2}\frac{a-c}{b-(N-1)}\ 𝟙$$
#
# The former expression justifies the following considerations:
# * the usage is the same for every consumer, as intuitively expected since in a fully connected graph everyone is equally influenced by everyone else;
# * the greater the difference between the sensibility to own consumption ($b$) and the effect of positive externality $(N-1)$, the smaller the usage of each consumer; in the limit $b \gg \sum_j g_{ij} = N-1$ the usage of each agent becomes negligible;
# * the conditions $a >c$ and $b > \sum_jg_{ij}$ guarantee that the consumers' usage is finite and positive definite.
#
#

# +
N = 9
a = 10.0
b = 12.0
c = 2.0

G = createGraphFullyConnected(N)

checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))

J = ones(N, N)
M_explicit = (b / (b + 1)) * I + (b / ((b + 1) * (b + 1 - N))) * J
x_star_explicit = (0.5 * (a - c) / (b - (N - 1))) * ones(N)
p_star_explicit = 0.5 * (a + c) * ones(N)

println("M teorica calcolata: ", round.(M_explicit, digits=4))
println("x_star teorica: ", round.(x_star_explicit, digits=4))
println("p_star teorica: ", round.(p_star_explicit, digits=4))

graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey)
# -

# ### Directed Ring
#
#
#
# We call *directed ring* a graph with $N$ nodes and the following connections:
#
# $$1 \rightarrow 2 \rightarrow 3 \rightarrow \dots \rightarrow N-1 \rightarrow N \rightarrow 1$$
#
# The adjacency matrix is:
#
# $$G = \begin{pmatrix} 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \ddots & \vdots \\ 0 & 0 & \cdots & 0 & 1 \\ 1 & 0 & \cdots & 0 & 0 \end{pmatrix}$$
#
# and it is orthogonal: $G^{-1} = G^\top$.
# We start by noticing that for a generic vector $y$:
#
# $$\begin{aligned}
#     (Gy)_i &= y_{i+1};\ (G^{-1}y)_i = (G^\top y)_i = y_{i-1}  \\
#     (G^ky)_i &= y_{i+k}; \ ((G^k)^{-1}y)_i = ((G^\top)^ky)_i=y_{i-k} \\
#    (G^Ny)_i &= y_{i+N} = y_i \  \Rightarrow \ \ G^N = I 
# \end{aligned}$$
#
# with the convention $N+1 \equiv 1$. Loosely speaking $G$ acts on a vector by shifting it 'forward', while $G^{-1}$ acts by shifting it 'backwards'.
# In particular:
#
# $$(G)^k𝟙 = 𝟙 \ ; \ (G^\top)^k𝟙 = 𝟙 \ \ \ \forall k$$
#
# Exploiting the condition $b > \sum_j g_{ij}$:
#
# $$\begin{aligned}
#     M𝟙 &=  \sum_{k=0}^{\infty} \left( \frac{G}{b}\right)^k 𝟙 = M^\top 𝟙 = \sum_{k=0}^{\infty} \left( \frac{G^\top}{b}\right)^k 𝟙= \sum_{k=0}^{\infty} \left( \frac{1}{b}\right)^k 𝟙 = \frac{b}{b-1}𝟙\\
#     &\Rightarrow (M+M^\top)𝟙 = \frac{2b}{b-1}𝟙\\
#     &\Rightarrow (M+M^\top)^{-1}𝟙 = \frac{b-1}{2b}𝟙
# \end{aligned}$$
#
# These properties are sufficient to compute the optimal price vector:
#
# $$\begin{aligned}
#     p^* &= c𝟙 + (a-c)(M + M^\top)^{-1} M 𝟙  = \left[c+(a-c)\frac{b}{b-1}(M+M^\top)^{-1}\right]𝟙 = \\
#     &= \left[c+(a-c)\frac{b}{b-1}\frac{b-1}{2b}\right]𝟙= \\
#     &= \left[c+\frac{1}{2}(a-c)\right]𝟙\\
#     &= \frac{1}{2}(a+c)𝟙
# \end{aligned}$$
#
# $p^*$ corresponds in turns to the same vector obtained for an undirected graph. This is not surprising, since despite being directed, the graph at hand is an example of *regular graph*, in which for each node the cardinalities of the out-neighborhood $|\mathcal{N}_i|$ and of the in-neighborhood $|\mathcal{N}_i^-|$ are equal and correspond to the same value for every node (in this case $|\mathcal{N}_i| = |\mathcal{N}_i^-| = 1 \ \forall i$), so that each node influences and is influenced by exactly one node.
#
# The consumers' usage Nash Equilibrium corresponding to the optimal price vector:
#
# $$x^* = M\frac{a𝟙-p^*}{b} = \frac{1}{2}\frac{a-c}{b}M𝟙 = \frac{1}{2}\frac{a-c}{b}\frac{b}{b-1}𝟙 = \frac{1}{2}\frac{a-c}{b-1}𝟙$$
#
# is analogous to the one obtained in the fully connected graph, considering that in this case $\sum_jg_{ij} = 1 \ \forall i$. Considerations similar to the ones for the fully connected graph can be done.

# +
N = 9
a = 10.0
b = 12.0
c = 2.0

G = createDirectedGraphRing(N)
checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))

x_star_explicit = (0.5 * (a - c) / (b - 1)) * ones(N)
p_star_explicit = 0.5 * (a + c) * ones(N)

println("x_star teorica: ", round.(x_star_explicit, digits=4))
println("p_star teorica: ", round.(p_star_explicit, digits=4))

using GraphRecipes
graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey, arrow=true)
# -

# ### Generic Regular Graph
#
# The previous examples suggest that the regularity of a graph plays a fundamental role in determining the optimal price vector and the consumers' usages at equilibrium. Extending the analysis to the case of weighted graphs, in which we can include in the adjacency matrix $G$ (now called weight matrix) different weights $g_{ij} \geq 0$ associated to different edges $(i,j)$, we say that a graph is *regular* if $\sum_jg_{ij}  = \sum_jg_{ji}  = w  \ \forall i$, i.e, if all nodes have the same out-degree, which is also equal to their in-degree. 
# In this cases, by hypothesis:
#
# $$\begin{aligned}
#     G𝟙 &= w𝟙 \\
#     G^\top𝟙 &= w𝟙
# \end{aligned}$$
#
# Then, if $b > \sum_j g_{ij} = w$:
#
# $$\begin{aligned}
#     M𝟙 &= \left(I-\frac{G}{b}\right)^{-1}𝟙 = \sum_{k=0}^\infty\left(\frac{G}{b}\right)^k𝟙 = \sum_{k=0}^\infty\left(\frac{w}{b}\right)^k𝟙 = \frac{b}{b-w}𝟙 \\
#     M^\top𝟙 &= \left[\left(I-\frac{G}{b}\right)^{-1}\right]^\top𝟙 = \sum_{k=0}^\infty\left[\left(\frac{G}{b}\right)^k\right]^\top𝟙 = \sum_{k=0}^\infty\left(\frac{w}{b}\right)^k𝟙 = \frac{b}{b-w}𝟙 
# \end{aligned}$$
#
# Therefore:
#
# $$\begin{aligned}
# p^* &= c𝟙 + (a-c)(M + M^\top)^{-1} M 𝟙 = \left[c+(a-c)\frac{b}{b-w}(M+M^\top)^{-1}\right]𝟙 = \\
#     &= \left[c+(a-c)\frac{b}{b-w}\frac{b-w}{2b}\right]𝟙= \\
#     &= \left[c+\frac{1}{2}(a-c)\right]𝟙\\
#     &= \frac{1}{2}(a+c)𝟙
# \end{aligned}$$
#
# and:
#
# \begin{align*}
# x^* & = M\frac{a𝟙-p^*}{b} =\frac{1}{2}\frac{a-c}{b}M𝟙 = 
# \frac{1}{2}\frac{a-c}{b}\frac{b}{b-w}𝟙 = 
# \frac{1}{2}\frac{a-c}{b-w}𝟙
# \end{align*}
#
# in perfect agreement with the fully connected example in which $w = N-1$ and with the directed ring example in which $w = 1$. Again, the conditions $a>c$ and $b>w$ are sufficient to guarantee a well defined usage inversely proportional to the difference $b-w$.
# Notice in all the cases discussed so far both the optimal prices and the Nash equilibrium usages are the same for every agent. 

# +
N = 9
a = 10.0
b = 12.0
c = 2.0
d = 4

G = adjacency_matrix(random_regular_graph(N, d))
G = Matrix(G) * 1.0

checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))

w = sum(G[1, :])
x_star_explicit = (0.5 * (a - c) / (b - w)) * ones(N)
p_star_explicit = 0.5 * (a + c) * ones(N)

println("x_star teorica: ", round.(x_star_explicit, digits=4))
println("p_star teorica: ", round.(p_star_explicit, digits=4))

graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey)
# -

# ### Undirected Star Graph
#
# In an *undirected star graph* a central node (or *hub*) is connected through undirected edges to $N-1$ leaves, which therefore are not connected to each other.
#
#
# Fixing the hub to be the node $i=1$ and the $N-1$ leaves to be the nodes $i=2\dots N$,
# the adjacency matrix is
# \begin{align*}
#     G = \begin{pmatrix} 0 & 1 & 1 & 1 &\cdots & 1 \\ 1 & 0 & 0 & 0 &\cdots & 0 \\1 & 0 & 0 & 0 &\cdots & 0 \\ \vdots & \vdots & \ddots & \ddots & \ddots &\vdots \\ 1 & 0 & 0 & 0 &\cdots & 0  \\1 & 0 & 0 & 0 &\cdots & 0  \end{pmatrix}
# \end{align*} 
#
# We immediately see that the graph is not regular, since 
# \begin{align*}
#     \sum_j g_{ij} = \begin{cases}
#         N-1 \ &\text{if $i=1$}\\
#         1 &\text{otherwise}
#     \end{cases}
# \end{align*}
# The graph is undirected, thus we can immediately deduce the optimal price vector $p^*=\frac{1}{2}(a+c)𝟙$.
# Recalling that in an undirected graph $x^* = \frac{1}{2}\frac{a-c}{b}M𝟙$ we focus on evaluating $z \doteq M𝟙 $. By definition
#
# \begin{align*}
#        z &= M𝟙 = \left(I-\frac{G}{b}\right)^{-1}𝟙 \\ \Rightarrow &(bI-G)z = b𝟙
# \end{align*}
# From which we get $N-1$ equivalent equations for the leaves:
# \begin{align*}
#        &bz_k-z_1 = b \\
#        \Rightarrow &z_k=\frac{b+z_1}{b} \  \ \forall k=2\dots N
# \end{align*}
# And an equation for the hub:
# \begin{align*}
#        bz_1-\sum_{k=2}^Nz_k = &b
# \end{align*}
# Combining them together we obtain:
# \begin{align*}
#         &bz_1- (N-1)\frac{b+z_1} {b} = b \\
#         &\Rightarrow \ \ z_1 = \frac{b (b+N-1)}{b^2-(N-1)}\\
#          &\Rightarrow \ \ z_k  = \frac{b(b+1)}{b^2-(N-1)}
# \end{align*}
# So that
# \begin{align*}
#     x^*=\frac{1}{2}\frac{a-c}{b} M 𝟙 =  \frac{1}{2}\frac{a-c}{b^2-(N-1)}\begin{bmatrix}
#     b+N-1 \\
#     b+1 \\
#     b+1\\
#     \vdots \\
#     b+1
# \end{bmatrix}
# \end{align*}    
#
# Despite the price being the same for every agent, the central hub is pushed to consume more than any leaf due to the great effect of positive externality.
#

# +
N = 9
a = 10.0
b = 12.0
c = 2.0

G = createUndirectedGraphStar(N)

checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))

z1 = b * (b + N - 1) / (b^2 - (N - 1))
zk = b * (b + 1) / (b^2 - (N - 1))
z = [z1; fill(zk, N-1)]
x_star_explicit = (0.5 * (a - c) / b) * z
p_star_explicit = 0.5 * (a + c) * ones(N)

println("x_star teorica: ", round.(x_star_explicit, digits=4))
println("p_star teorica: ", round.(p_star_explicit, digits=4))

graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey)
# -

# ### Directed Star Graph: Single Influencer and Isolated Followers
#
# We recall that, by looking at the best response for each agent $ B_i(x_{-i})= \frac{a-p_i}{b}+\frac{1}{b}\sum_jg_{ij}x_j$, we must interpret a directed edge $(i,j)$ from $i$ to $j$ as the influence that $j$ has on $i$. It follows that to model a star graph with a central influencer we must construct a graph in which all the leaves in the star have an outgoing edge connecting them to the central hub. Namely:
# \begin{align*}
# G = \begin{pmatrix} 0 & 0 & 0 &\cdots & 0 \\ 1 & 0 & 0 &\cdots & 0  \\ 1 & 0 & 0 &\cdots & 0 \\ \vdots & \ddots & \ddots & \ddots  &\vdots \\1 & 0 & 0 &\cdots & 0  \end{pmatrix}
# \end{align*}
# We immediately deduce: 
# \begin{align*}
#     &G^k = 0 \ \ \forall k >1 \\
# \end{align*}
# since no paths of length $l>1$ are present in the graph (the hub is a sink). 
# Moreover
# \begin{align*}
#     &G𝟙 = 𝟙-\delta^{(1)}\\
# \end{align*}
# Thus 
# \begin{align*}
#        &M= \sum_{k=0}^\infty\left(\frac{G}{b}\right)^k = \sum_{k=0}^1\left(\frac{G}{b}\right)^k = \left(I+\frac{G}{b}\right) \\
#        &M𝟙 = 𝟙+\frac{1}{b}(𝟙-\delta^{(1)}) = \frac{b+1}{b}𝟙-\frac{1}{b}\delta^{(1)} = \begin{bmatrix}
#         1 \\
#         (b+1)/b \\
#         (b+1)/b\\
#         \vdots \\
#         (b+1)/b\\
#         \end{bmatrix} \\
#        &M+M^T = 2I+\frac{1}{b}(G+G^\top)\ = 2I + \frac{1}{b}G_{und} = \begin{pmatrix} 2 & 1/b & 1/b &\cdots & 1/b \\ 1/b & 2 & 0 &\cdots & 0  \\ 1/b & 0 & 2 &\cdots & 0 \\ \vdots & \ddots & \ddots & \ddots  &\vdots \\1/b & 0 & 0 &\cdots & 2
#     \end{pmatrix}
# \end{align*}
# with $G_{und}$ being the adjacency matrix of the undirected star graph.
# To find the optimal price vector, we're interested in computing $\phi \doteq (M+M^\top)^{-1}M𝟙$: this is achieved by solving the system
# \begin{align*}
#     (M+M^\top)\phi = M𝟙 \iff 
#     \begin{pmatrix} 2 & 1/b & 1/b &\cdots & 1/b \\ 1/b & 2 & 0 &\cdots & 0  \\ 1/b & 0 & 2 &\cdots & 0 \\ \vdots & \ddots & \ddots & \ddots  &\vdots \\1/b & 0 & 0 &\cdots & 2
#     \end{pmatrix}\begin{bmatrix}
#         \phi_1 \\
#         \phi_2 \\
#         \phi_3\\
#         \vdots \\
#         \phi_N\\
#         \end{bmatrix} = 
#     \begin{bmatrix}
#         1 \\
#         (b+1)/b \\
#         (b+1)/b\\
#         \vdots \\
#         (b+1)/b\\
#         \end{bmatrix}
# \end{align*}
# By symmetry of the problem (all leaves are equivalent): $\phi_2=\phi_3 = \dots = \phi_N \doteq \phi_L$. The system reduces to:
# \begin{align*}
#     &2\phi_1 + \frac{N-1}{b}\phi_L = 1 \\ 
#     &\frac{\phi_1}{b} + 2\phi_L = \frac{b+1}{b}\\
#     \Rightarrow \ &\phi = (M+M^\top)^{-1}M𝟙= \frac{1}{4b^2-(N-1)}\begin{bmatrix}
#         2b^2-(b+1)(N-1) \\
#         2b^2+b \\
#         2b^2+b\\
#         \vdots \\
#         2b^2+b\\
#         \end{bmatrix} 
# \end{align*}
# So that finally:
# \begin{align*}
# p^* &= c𝟙 + (a-c)(M + M^\top)^{-1} M 𝟙 = c𝟙+(a-c)\phi = \\
#     &= \begin{bmatrix}
#         c \\
#         c \\
#          c\\
#         \vdots \\
#          c\\
#         \end{bmatrix}+\frac{a-c}{4b^2-(N-1)}\begin{bmatrix}
#         2b^2-(b+1)(N-1) \\
#         2b^2+b \\
#         2b^2+b\\
#         \vdots \\
#         2b^2+b\\
#         \end{bmatrix} 
# \end{align*}
# It is indeed verified that for $b>(N-1)$, $p^*_1 < p^*_L$, confirming the fact that the hub, i.e. the agent that strongly influence the others, gets a lower price.\
# It can be meaningful to investigate the limit $b \gg \sqrt{(N-1)} \iff \frac{(N-1)}{b^2} \ll 1$, where the sensibility to own consumption (actually, its square) is much greater than the positive externality effect. In this case
# \begin{align*}
#     &\phi_1 = \frac{2-\frac{(N-1)}{b}-\frac{(N-1)}{b^2}}{4-\frac{(N-1)}{b^2}} = \frac{1}{2} -\frac{(N-1)}{4b} + o\left(\frac{(N-1)}{b^2}\right) \Rightarrow p^*_1 \simeq \frac{1}{2}(a+c)-\frac{a-c}{b}(N-1) \\
#     &\phi_L = \frac{2+\frac{1}{b}}{4-\frac{(N-1)}{b^2}} = \frac{1}{2}+\frac{1}{4b} + o\left(\frac{(N-1)}{b^2}\right) \Rightarrow p^*_L \simeq \frac{1}{2}(a+c)+\frac{a-c}{4b} 
# \end{align*}
# The price applied on the hub is smaller than the unbiased one, while
# the price applied on leaves has still a positive increment $\frac{a-c}{4b}$. This corrections becomes negligible when $b \gg (a-c)$ and $b \gg (N-1)$. \
# Finally, the corresponding consumers' usages at Nash equilibrium are
# \begin{align*}
#     x^* &= M \frac{a𝟙 - p^*}{b} = \frac{a-c}{b}M[𝟙-\phi] = \frac{a-c}{b^2}\begin{bmatrix}
#         b(1-\phi_1) \\
#         (1-\phi_1)+b(1-\phi_L) \\
#         (1-\phi_1)+b(1-\phi_L)\\
#         \vdots \\
#         (1-\phi_1)+b(1-\phi_L)\\
#         \end{bmatrix}\\
# \end{align*}
# and $x_1^* > x_L^*$ holds: the central hub consumes more than anyone else.

# +
N = 9
a = 10.0
b = 12.0
c = 2.0

G = createDirectedGraphStar(N)

checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))

denom = 4 * b^2 - (N - 1)
phi1 = (2 * b^2 - (b + 1) * (N - 1)) / denom
phil = (2 * b^2 + b) / denom
phi = [phi1; fill(phil, N-1)]

x_explicit = (a - c) / b^2 * [b * (1 - phi1); fill((1 - phi1) + b * (1 - phil), N-1)]
println("x_star teorica: ", round.(x_explicit, digits=4))

p_star_explicit = c * ones(N) + (a - c) * phi
println("p_star teorica: ", round.(p_star_explicit, digits=4))


graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey, arrow=true)
# -

# ### Single Influencer and Fully Connected Followers
#
# With respect to the previous case we now construct a network in which nodes $i = 2\dots N$ form a fully connected graph and from each of these nodes there's an outgoing edge connecting it to a hub $i=1$. Namely:
# \begin{align*}
#     G = \begin{pmatrix} 0 & 0 & 0 &\cdots & 0 \\ 1 & 0 & 1  &\cdots & 1 \\1 & 1 & 0 &\cdots & 1 \\ \vdots & \vdots & \ddots & \ddots &\vdots \\ 1 & 1 & 1  &\cdots & 0  \\ \end{pmatrix}
# \end{align*}
# For the sake of semplicity we directly provide the result of $\phi = (M+M^T)^{-1}M𝟙$
# \begin{align*}
#     \phi = \begin{bmatrix}
# \phi_1 \\[8pt]
# \phi_L\\
# \vdots \\
# \phi_L
# \end{bmatrix} = \frac{1}{4b^2 - 4b(N-2) - (N-1)}
# \begin{bmatrix}
# 2b^2 - b(3N-5) - (N-1) \\[8pt]
# (2b+1)(b-(N-2))\\
# \vdots \\
# (2b+1)(b-(N-2))
# \end{bmatrix}
# \end{align*}
# with the optimal price being
# \begin{align*}
#     p^* = c𝟙 + (a-c)\phi 
# \end{align*}
# Again the price applied on the hub is lower. Comparing this result with the one obtained in the directed star we're interested in showing how the network between the followers has influenced their price
# \begin{align*}
#    p^*_{\text{connected follower}} - p^*_{\text{isolated follower}}   =  \frac{(2b+1)(b-(N-2))}{4b^2 - 4b(N-2) - (N-1)} -\frac{2b^2+b}{4b^2-(N-1)}  > 0 \ \ \  \forall b, N |\  b >N-1, N>2
# \end{align*}
# That is to say the positive externality added in this second case (if we're in a well posed problem with $N>2$) causes the  price on the followers to rise.
# Finally:
# \begin{align*}
#     x^* = \frac{a-c}{b}M[𝟙-\phi] = \frac{a-c}{b}\begin{bmatrix}
# 1-\phi_1 \\[8pt]
# \dfrac{b}{b-(N-2)}\left( \dfrac{1-\phi_1}{b}+1-\phi_L\right) \\
# \vdots \\
# \dfrac{b}{b-(N-2)}\left( \dfrac{1-\phi_1}{b}+1-\phi_L\right)
# \end{bmatrix}
# \end{align*}
# This time it is verified that, for a well posed problem,
# $x^*_1 < x^*_L$: not only the central hub gets a price lower than the price for the leaves, it is also pushed to consume less than the others.

# +
N = 9
a = 10.0
b = 12.0
c = 2.0

G = zeros(N, N)
G[2:N, 2:N] = createGraphFullyConnected(N-1)
G[2:N, 1] .= 1.0

checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))

denom = 4*b^2-4*b*(N-2)-(N-1)
phi1 = (2*b^2-b*(3*N-5)-(N-1))/ denom
phil = ((2*b+1)*(b-(N-2))) / denom
phi = [phi1; fill(phil, N-1)]


x_star_explicit = (a - c) / b * [(1 - phi1); fill((b)/(b-(N-2))*((1 - phi1)/b + 1 - phil), N-1)]
println("x_star teorica: ", round.(x_star_explicit, digits=4))

p_star_explicit = c * ones(N) + (a - c) * phi
println("p_star teorica: ", round.(p_star_explicit, digits=4))



denom = 4 * b^2 - (N - 1)
phil_isolated = (2 * b^2 + b) / denom
p_star_isolated = c + (a - c) * phil_isolated

pl_diff = p_star[2] - p_star_isolated


println("Prezzo_connesso - Prezzo_isolato: ", round(pl_diff, digits=4))

graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey)
# -

# ### Multiple Isolated Influencers and Communities of Followers
#
# We construct a graph with $N_{inf}$ isolated influencers and the remaining nodes organized in communities of $S_{com}$ nodes each.
#
#    

# +
N_infl = 1
S_com = 4
n_com = 3;
N = N_infl + n_com*S_com
a = 10.0
b = 12.0
c = 2.0

G = createGraphInfluencersCommunities(N_infl, S_com, n_com)

checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))




graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey)
# -

# Può essere utile confronare il caso con un influencer e due comunità da 4 vs il caso di prima con 1 influencer e 8 fully connected

# INFLUENCER DEGLI INFLUENCERS E COMUNITà

# +
N_infl = 3
S_com = 3
n_com = 3;
N = N_infl + n_com*S_com + 1
a = 10.0
b = 12.0
c = 2.0

G = zeros(N, N)
G[2:N_infl+1, 1] .= 1.0
G[2:N, 2:N] = createGraphInfluencersCommunities(N_infl, S_com, n_com)
G[2:N_infl+1, 2: N_infl+1] =createGraphFullyConnected(N_infl)

checkParameters(G, a, b, c)

M = influenceMatrix(G, b)

x_star = bestResponse(M, a, b, c)
println("Equilibrio Nash per gli usi: ", round.(x_star, digits=4))

p_star = bestPrice(M, a, b, c)
println("Prezzo ottimale: ", round.(p_star, digits=4))




graphplot(G, names=1:N, nodesize=0.3, curves=false, markercolor=:lightgrey)
# -

# ## Continuous-time Dynamics and Optimal Control
#
# Let's consider the adoption dynamics due to the following continuous-time relaxation towards best response:
#
# $$\begin{aligned}
# \dot{x}(t) &= -\Lambda x(t) + \frac{1}{b}(a𝟙 - p(t) + Gx(t)) = \\
# &= \left( \frac{1}{b} G - \Lambda \right) x(t) + \frac{1}{b}(a𝟙 - p(t)) \doteq \\
# &\doteq D x(t) + f(t)
# \end{aligned}$$
#
# where $\Lambda=\text{diag}(\pi_i)$ is an individual update rate and we defined $D \doteq \frac{1}{b} G - \Lambda$ and $f(t) \doteq \frac{1}{b}(a𝟙 - p(t))$. 
#
# To solve this differential equation, we use the ansatz $x(t) = e^{D t} c(t)$, where $c(t)$ is an unknown function depending on $t$. We get:
#
# $$\begin{aligned}
#     \dot{x}(t) &= D e^{D t} c(t) + e^{D t} \dot{c}(t) \\
#      D x(t) + f(t) &= D e^{D t} c(t) + e^{D t} \dot{c}(t) \\
#     \dot{c}(t) &=  e^{-D t} f(t) \\
#      \Rightarrow  c(t) &= c(0) + \int_0^t \mathrm{d}\tau \: e^{-D \tau} f(\tau)
# \end{aligned}$$
#
# where $c(0)=x(0)$. So the solution is:
#
# $$x(t) = e^{D t} x(0) + \int_0^t \mathrm{d}\tau \: e^{D(t- \tau)} f(\tau)$$
#
# Suppose there is a cost for price variation and introduce the latter as a control variable $u = \dot{p}$. The price control problem can be made linear-quadratic defining the augmented state:
#
# $$z = \begin{pmatrix} x \\ p \end{pmatrix}$$
#
# which undergoes the linear dynamics:
#
# $$\begin{aligned}
#     \dot{z} &= \begin{pmatrix} -\Lambda + \frac{1}{b}G & -\frac{1}{b}I \\ 0 & 0 \end{pmatrix} z + \begin{pmatrix} 0 \\ I \end{pmatrix} u + \begin{pmatrix} \frac{a}{b}𝟙 \\ 0 \end{pmatrix} \doteq \\
#     &\doteq Az+Bu+d
# \end{aligned}$$
#
# where we defined $A \doteq \begin{pmatrix} -\Lambda + \frac{1}{b}G & -\frac{1}{b}I \\ 0 & 0 \end{pmatrix}$, $B \doteq \begin{pmatrix} 0 \\ I \end{pmatrix}$ and $d \doteq \begin{pmatrix} \frac{a}{b}𝟙 \\ 0 \end{pmatrix}$. The intertemporal revenue becomes a quadratic form in the augmented space:
#
# $$R = x^\top p = \frac{1}{2} z^\top Q z, \quad \text{with} \quad Q = \begin{pmatrix} 0 & I \\ I & 0 \end{pmatrix}$$
#
# so that the overall objective function of the monopolist becomes:
#
# $$\mathcal{R} = \max_{u(\cdot)} \int_{0}^{T} e^{-\delta t} \left[ \frac{1}{2} z^\top Q z - \frac{1}{2} u^\top K u \right] dt$$
#
# where $K = \kappa I$ represents the cost of price variation and $u = \dot{p}$ is the control variable. 
#
# Let's define the current cost $L(x,u,t)$ as:
#
# $$L(x,u,t) \doteq e^{-\delta t} \left[- \frac{1}{2} z^\top Q z + \frac{1}{2} u^\top K u \right]$$
#
# so we can rewrite this problem as a minimization problem:
#
# $$\mathcal{V} = -\mathcal{R} = \min_{u(\cdot)} \int_{0}^{T} L(x,u,t) dt$$
#
# The PMP provides necessary conditions for optimality. However, in this linear-quadratic problem, since the dynamics are linear, the objective function is concave in $u$ and the condition $b > \sum_j g_{ij} \quad \forall i$ prevents the state variables $(x, p)$ from diverging, the first-order conditions derived from the PMP are also sufficient for a global maximum.
#
# We define the Hamiltonian:
#
# $$\begin{aligned}
#     H(x,u,t) &= L(x,u,t) + \lambda(t)^\top \left[ Az+Bu+d \right] = \\
#     &= e^{-\delta t} \left[- \frac{1}{2} z^\top Q z + \frac{1}{2} u^\top K u \right] + e^{-\delta t} \left( \lambda(t)^\top e^{\delta t} \right)   \left[ Az+Bu+d \right] \doteq \\
#     &\doteq e^{-\delta t} \ \mathcal{H}(x,u,t)
# \end{aligned}$$
#
# where we defined the Hamiltonian without time discounting $\mathcal{H}(x,u,t) \doteq  \left[- \frac{1}{2} z^\top Q z + \frac{1}{2} u^\top K u \right] +  \mu(t)^\top   \left[ Az+Bu+d \right]$, where $\mu(t) \doteq \lambda(t) e^{\delta t}$. Applying the PMP, we obtain the following system of equations:
#
# $$\begin{cases}
#     \dot{z} &= \nabla_\lambda H  \\ 
#     \dot{\lambda} &= - \nabla_z H \\
#     0 &= \nabla_u H
# \end{cases}$$
#
# with $z(0)=z_0$ and $\lambda(T)=0$. So we get:
#
# $$\begin{cases}
#     \dot{z} &= Az+Bu+d \\
#     \dot{\lambda} &= e^{-\delta t}  \left( Qz - A^\top \mu  \right) \\
#     \nabla_u H &= e^{-\delta t}  \left(Ku + B^\top \mu \right) = 0 \implies u(t) = -K^{-1} B^\top \mu(t)
# \end{cases}$$
#
# From $\lambda(t) = \mu(t) e^{-\delta t}$ it follows $\dot{\lambda}(t) = \dot{\mu}(t) e^{-\delta t} - \delta \mu(t) e^{-\delta t}$, and we get the equation:
#
# $$\begin{aligned}
#     e^{-\delta t}  \left( Qz - A^\top \mu  \right) &= \dot{\mu} \, e^{-\delta t} - \delta \, \mu e^{-\delta t} \\
#     \dot{\mu}(t) &=  \left(  - A^\top +\delta \, \mathrm{I}  \right) \mu (t) + Qz(t)
# \end{aligned}$$
#
# We use now the ansatz $\mu (t) = S(t) z(t) + \nu (t)$, where $S(t)\in \mathbb{R}^{2N\times 2N}, \ \nu(t)\in \mathbb{R}^{2N}$. Deriving with respect to $t$ we get:
#
# $$\begin{aligned}
#     \dot{\mu} &= \dot{S}z+S \dot{z} +\dot{\nu} = \\
#     &=  \dot{S}z + S(Az + Bu + d) + \dot{\nu} = \\
#     &= \dot{S}z + S(Az - BK^{-1}B^\top\mu + d) + \dot{\nu} = \\
#     &= \dot{S}z + SAz - SBK^{-1}B^\top Sz - SBK^{-1}B^\top \nu + Sd + \dot{\nu}
# \end{aligned}$$
#
# Comparing with the previous expression we get:
#
# $$\begin{aligned}
#       \dot{S}z + SAz - SBK^{-1}B^\top Sz - SBK^{-1}B^\top \nu + Sd + \dot{\nu} &= \left(  - A^\top +\delta \, \mathrm{I}  \right) ( S z + \nu) + Qz \\
#     \left[ \dot{S} + A^\top S + SA - \delta S - SBK^{-1}B^\top S - Q  \right] z &= -\dot{\nu}+ \left[SBK^{-1}B^\top - A^\top + \delta I \right] \nu - Sd
# \end{aligned}$$
#
# This must be valid $\forall z$, so it follows that:
#
# $$\begin{cases}
# \dot{S} &= -A^\top S - SA + \delta S + SBK^{-1}B^\top S + Q \\
# \dot{\nu} &= (SBK^{-1}B^\top - A^\top + \delta I)\nu - Sd
# \end{cases}$$
#
# From $\lambda(T)=0$ it follows $\mu(T)=0$. So the boundary conditions are $S(T)z(T)+\nu(T) =0$, and because it has to be valid $\forall \, z(T)$, it follows that $S(T)=0, \ \nu(T)=0$.
#
# We can integrate these equations backward in time to find $S(t), \ \nu(t)$, and finally we have the $2$ equations for $\dot{x}(t), \ \dot{p}(t)$ assuming optimal control:
#
# $$\begin{cases}
# \dot{x}(t) &= -\Lambda x(t) + \frac{1}{b}(a𝟙 - p(t) + Gx(t))\\
# u(t) &= \dot{p}(t) = -K^{-1} B^\top \left( S(t) z(t) + \nu(t) \right)
# \end{cases}$$

# +
include("myfunctions.jl")

𝛿 = 1.0             #time discount

"""INTEGRATOR TIME STEPS"""
T_0 = 0.0
T_F = 20.0
Δt = 0.01

times = Vector(T_0:Δt:T_F)

steps = length(times)

"""MATRICES"""

#λ = G/b
#Λ = diagm(vec(sum(G/b;dims=2)))
Λ = 0.2*diagm(ones(N)+0.25*rand(N)-0.25*rand(N))

k = 1.0
K = k*diagm(ones(N))

# Assicuriamoci che G sia una matrice e b uno scalare
G_matrix = G # Se 'g' è il grafo creato con Graphs.jl
block_top_left = Λ * ( (G_matrix ./ b) - I )

block_top_left = Λ * (G_matrix/b - I)  
block_top_right = -Λ/b
block_bottom_left = zeros(N, N)
block_bottom_right = zeros(N, N)

A = [block_top_left  block_top_right;
     block_bottom_left block_bottom_right]

B = [zeros(N, N); I(N)]


d_vec = vcat(Λ * (a/b) * ones(N), zeros(N))


"""
A = [Λ*(G/b - I) (-Λ/b);
     0I(N)      0I(N)]

B = [0I(N); I(N)]

D = vcat(Λ*(a/b)*ones(N), zeros(N));


g = c*vcat(ones(N), zeros(N));
"""
#display(A)
# -

include("myfunctions.jl")
g = c*vcat(ones(N), zeros(N));
S,v = integrate_backward(A, B, d_vec, g, K, N, 𝛿, Δt, steps);

# +
include("myfunctions.jl")

#initial condition near NE
x0 = x_star.*(ones(N)+0.05*rand(N)-0.05*rand(N))
p0 = p_star.*(ones(N)+0.05*rand(N)-0.05*rand(N))

#x0 = rand(N) + 10*ones(N)
#p0 = rand(N).+4

println("INITIAL CONDITIONS")
println("x0: $x0")
println("p0: $p0")

X, P, U = integrate_forward(x0, p0, S, v, G_matrix, K, Λ, B, N, a, b, c, Δt, steps)

println("END OF INTEGRATION")
println("Final usages: $(X[:,steps])")
println("Final prices: $(P[:,steps])")
# -

# ### VISUALIZATION
#
# Since continuous time did not work I'll create some plots oscillating around NE

# +
p = plot()

for i in 1:N
    plot!(times, X[i,:], label="x_$i")
end
title!("Consumers' usage")
xlims!(T_0, T_F)
xlabel!("t")
ylabel!("x")

# +
p = plot()

for i in 1:N
    plot!(times, P[i,:], label="p_$i")
end
title!("Price Target")
xlims!(T_0, T_F)
xlabel!("t")
ylabel!("p")

# +
my_reward = calculate_reward(times, X, P, U, c, 𝛿, k)

plot(times, my_reward, label="cumulated reward")

title!("Reward")
xlims!(T_0, T_F)
xlabel!("t")
ylabel!("reward")
# -

# ## Network Pricing with Asynchronous Noisy Best Response
#
#
# ### 1. Consumer Utility and Noisy Best Response 
#
# Let's consider a network of $N$ consumers. The utility of consumer $i$ is
# $$u_i(x_i) = ax_i - \frac{b}{2}x_i^2 + \left( \sum_{j=1}^N g_{ij}x_j \right)x_i - p_i x_i$$
#
#
# Consumers are boundedly rational. When evaluating their optimal adoption level, they follow a Noisy Best Response rule, where the probability of choosing $x_i$ is proportional to the exponential of their utility, weighted by a rationality parameter $\beta$:
# $$P(x_i) \propto \exp\{ \beta \cdot u_i(x_i, x_{-i}) \}$$
#
# Because the utility function is quadratic with respect to $x_i$, we can rewrite it as $u_i = -\frac{b}{2}x_i^2 + E_i x_i$, where $E_i = a - p_i + \sum g_{ij}x_j$. By substituting this into the probability distribution we get
# $$P(x_i) \propto \exp\left\{ -\frac{\beta b}{2} \left(x_i - \frac{E_i}{b}\right)^2 \right\}$$
# So the NBR assumes the same form as the probability density function of a Gaussian Distribution $\mathcal{N}(\mu_i, \sigma^2)$, where
# * Mean: $\mu_i = \frac{a - p_i + \sum g_{ij}x_j}{b}$
# * Variance: $\sigma^2 = \frac{1}{\beta b}$
#
# As $\beta \to \infty$ , the variance approaches zero, and the NBR converges into the deterministic Best Response.
#
# ### 2. Asynchronous Network Dynamics 
#
# We distinguish between macro-time, that is when the monopolist adjust the price, and the micro-time between 2 macro-time steps, that is when consumers adjust their consumption. At each micro-time step, a random consumer $i$ is selected to change $x_i$.
#
# We introduce a resistance to change parameter, $s_i \in [0, 1]$. 
# * With probability $s_i$, the consumer ignores the update opportunity and keeps their current adoption: $x_i(t+1) = x_i(t)$.
# * With probability $1 - s_i$, the consumer updates their adoption using the NBR distribution: $x_i \sim \mathcal{N}(\mu_i, \sigma^2)$.
#
# ### 3. Greedy Bellman Equation
#
# The monopolist aims to find the optimal price variation $\Delta p_i$ to maximize expected profit at the next macro-time step. For this, we use a greedy Bellman Equation with a time horizon $T=1$. The reward function for the monopolist includes a penalty $\kappa (\Delta p_i)^2$ to prevent rapid changes in prices.
# $$\max_{\Delta p_i} \mathbb{E}[r_{i,t}] = \max_{\Delta p_i} \left[ (p_i + \Delta p_i - c) \cdot \mathbb{E}[x_{i, t+1}] - \kappa (\Delta p_i)^2 \right]$$
#
# The monopolist estimates the expected response of node $i$ by assuming the rest of the network remains frozen at the current state $x(t)$. So between each marco-time step, consumers adjust their consumption asinchronously according to NBR dynamics, considering the consuption the rest of the network fronzen at the previous macro-time step.
#
# The expected adoption is
# $$\mathbb{E}[x_{i, t+1}] = s_i x_{i,t} + (1-s_i) \frac{a-(p_i+\Delta p_i) + \sum_j g_{ij} x_{j,t}}{b}$$
#
# We define two auxiliary variables 
# * $A_i = s_i x_i + (1-s_i) \left( \frac{a - p_i + \sum g_{ij}x_j}{b} \right)$, that represents the expected adoption if the price remains unchanged
# * $B_i = \frac{1-s_i}{b}$, that represents the sensitivity of expected adoption to price changes
# So the expected adoption becomes
# $$\mathbb{E}[x_{i, t+1}] = A_i - B_i \Delta p_i $$
#
# Substituting this into Bellman Equation gives
# $$\max_{\Delta p_i} \mathbb{E}[r_{i,t}] = \max_{\Delta p_i} \left[ (p_i + \Delta p_i - c) \cdot (A_i - B_i \Delta p_i) - \kappa (\Delta p_i)^2 \right]$$
#
# To find the global maximum, we take the first derivative with respect to $\Delta p_i$ and set it to zero
# $$ -2(B_i + \kappa)\Delta p_i^* + [A_i - B_i(p_i - c)] = 0$$
#
# Solving for $\Delta p_i^*$, we obtain
# $$\Delta p_i^* = \frac{A_i - B_i(p_i - c)}{2(B_i + \kappa)}$$

# +
using LinearAlgebra
using Distributions
using Graphs, GraphRecipes, Plots

# ==================================================================
# 1. SOLUZIONE ESATTA BELLMAN (ORIZZONTE H=1)
# ==================================================================
function exact_greedy_bellman(x, p, G, a, b, c, s, kappa)
    return ((s .* x .+ (1.0 .- s) .* (a .- p .+ G * x) ./ b) .- (1.0 .- s) ./ b .* (p .- c)) ./ (2.0 .* ((1.0 .- s) ./ b .+ kappa))
end

# ==================================================================
# 2. DINAMICA DEGLI UTENTI: NOISY BEST RESPONSE
# ==================================================================
# Simula la reazione degli utenti in base ai nuovi prezzi
function simulate_users!(x, p, G, a, b, beta, s, steps)
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

# ==================================================================
# MAIN: ESEMPIO DI ESECUZIONE
# ==================================================================
# Il monopolista e gli utenti evolvono per più step macro e micro.
# Parametri di base
T = 10                    # numero di macro-step temporali
N = 10                    # numero di nodi richiesto
a = 10.0
b = 2.0
c = 1.0
beta = 5.0
kappa = 0.5
s = fill(0.3, N)           # vettore resistenze uniforme

# ==================================================================
# Costruzione di un grafo orientato
# ==================================================================
# utilizziamo una matrice di adiacenza pesata
G = zeros(Float64, N, N)
for i in 1:N, j in 1:N
    if i != j
        # i primi due nodi hanno probabilità maggiore di inviare un arco
        if rand() < (i <= 2 ? 0.6 : 0.15)
            G[i, j] = rand() * 0.3
        end
    end
end

# visualizziamo il grafo con graphplot (versione orientata)
g = DiGraph(G .> 0)                                    # struttura per graphplot
# selezioniamo teoricante il backend grafico in caso non sia impostato
gr()
plt = graphplot(g, names=1:N, arrowsize=0.3, nodesize=0.3, method=:spring,
          directed=true, markercolor=:lightgrey)
# mostriamo il plot teoricante
display(plt)

println("Grafo orientato")
println("Out-degrees: ", sum(G .> 0, dims=2)[:])
println("In-degrees:  ", sum(G .> 0, dims=1)[:])

# inizializziamo le adozioni e i prezzi
x = rand(N)
p = fill(3.0, N)

# memorizziamo storia per poterla analizzare in seguito (facoltativo)
history_x = zeros(T, N)
history_p = zeros(T, N)

# evoluzione: T macro‑step, ciascuno con 3·N micro‑step
for t in 1:T
    println("\n=== Macro step ", t, " ===")
    println("stato iniziale x: ", round.(x, digits=3))
    println("stato iniziale p: ", round.(p, digits=3))

    # monopolista calcola variazione ottima
    delta_p = exact_greedy_bellman(x, p, G, a, b, c, s, kappa)
    p .+= delta_p

    # micro‑aggiornamenti degli utenti
    simulate_users!(x, p, G, a, b, beta, s, 3 * N)

    println("stato finale x: ", round.(x, digits=3))
    println("stato finale p: ", round.(p, digits=3))

    history_x[t, :] = x
    history_p[t, :] = p
end

# ==================================================================
# PLOT: Evoluzione temporale di x e p per nodi interessanti
# ==================================================================
# Identifichiamo i nodi più interessanti: quelli con gradi (out+in) più alti
out_degrees = sum(G .> 0, dims=2)[:]
in_degrees = sum(G .> 0, dims=1)[:]
total_degrees = out_degrees .+ in_degrees

# Selezioniamo i 3 nodi con degree totale più alto
top_nodes = sort(1:N, by=i->total_degrees[i], rev=true)[1:3]

println("\n=== Nodi più interessanti (per grado) ===")
for node in top_nodes
    println("Nodo $node: out-degree=$(out_degrees[node]), in-degree=$(in_degrees[node]), total=$(total_degrees[node])")
end

# Creiamo i plot per x e p in funzione di t
t_steps = 1:T
plt_x = plot(legend=:topright, xlabel="Macro-step", ylabel="Adozione x", title="Evoluzione delle adozioni", lw=2)
plt_p = plot(legend=:topright, xlabel="Macro-step", ylabel="Prezzo p", title="Evoluzione dei prezzi", lw=2)

colors = [:red, :blue, :green]
for (idx, node) in enumerate(top_nodes)
    plot!(plt_x, t_steps, history_x[:, node], label="Nodo $node", color=colors[idx], marker=:circle)
    plot!(plt_p, t_steps, history_p[:, node], label="Nodo $node", color=colors[idx], marker=:circle)
end

# Visualizziamo i plot
p_combined = plot(plt_x, plt_p, layout=(1, 2), size=(1200, 400))
display(p_combined)

# +
# ==================================================================
# SIMULAZIONE CON GRAFO INDIRETTO
# ==================================================================
using LinearAlgebra, Distributions, Graphs, GraphRecipes, Plots, SparseArrays

println("\n" * "="^60)
println("SIMULAZIONE CON GRAFO INDIRETTO - T=10, N=10")
println("="^60)

# Parametri
T = 50                   # numero di macro-step temporali
N = 10
a = 10.0
b = 2.0
c = 1.0
beta = 5.0
kappa = 0.5
s = fill(0.3, N)

# ==================================================================
# Costruzione di un grafo indiretto casuale (con matrice simmetrica)
# ==================================================================
# Creiamo una matrice di adiacenza simmetrica per grafo indiretto
G_undirected = zeros(Float64, N, N)
for i in 1:N
    for j in (i+1):N
        if rand() < 0.3  # probabilità di connessione
            weight = rand() * 0.4
            G_undirected[i, j] = weight
            G_undirected[j, i] = weight  # rendo simmetrica
        end
    end
end

println("\nMatrice di adiacenza (simmetrica):")
println("Sparsità: ", round(nnz(sparse(G_undirected)) / (N^2) * 100, digits=2), "%")

# Visualizziamo il grafo indiretto
g_undirected = Graph(G_undirected .> 0)
gr()
plt_undirected = graphplot(g_undirected, names=1:N, nodesize=0.3, method=:spring,
                           markercolor=:lightgrey, curves=false)
display(plt_undirected)

degrees = degree(g_undirected)
println("\nGradi dei nodi (grafo indiretto):")
for i in 1:N
    println("Nodo $i: grado=$(degrees[i])")
end

# ==================================================================
# EVOLUZIONE
# ==================================================================
# Inizializziamo adozioni e prezzi
x_und = rand(N)
p_und = fill(3.0, N)

# Matrice storia
history_x_und = zeros(T, N)
history_p_und = zeros(T, N)

# Ciclo di evoluzione: T macro-step
for t in 1:T
    println("\n=== Macro step $t ===")
    println("stato iniziale x: ", round.(x_und, digits=3))
    println("stato iniziale p: ", round.(p_und, digits=3))

    # Monopolista calcola variazione di prezzo
    delta_p = exact_greedy_bellman(x_und, p_und, G_undirected, a, b, c, s, kappa)
    p_und .+= delta_p

    # Micro-aggiornamenti degli utenti
    simulate_users!(x_und, p_und, G_undirected, a, b, beta, s, 3 * N)

    println("stato finale x: ", round.(x_und, digits=3))
    println("stato finale p: ", round.(p_und, digits=3))

    history_x_und[t, :] = x_und
    history_p_und[t, :] = p_und
end

# ==================================================================
# PLOT: Evoluzione per i 3 nodi con grado più alto
# ==================================================================
top_degree_nodes = sort(1:N, by=i->degrees[i], rev=true)[1:3]

println("\n=== Nodi con grado più alto (grafo indiretto) ===")
for node in top_degree_nodes
    println("Nodo $node: grado=$(degrees[node])")
end

# Plot temporali
t_steps = 1:T
plt_x_und = plot(legend=:topright, xlabel="Macro-step", ylabel="Adozione x", 
                 title="Evoluzione adozioni (grafo indiretto)", lw=2)
plt_p_und = plot(legend=:topright, xlabel="Macro-step", ylabel="Prezzo p", 
                 title="Evoluzione prezzi (grafo indiretto)", lw=2)

colors = [:red, :blue, :green]
for (idx, node) in enumerate(top_degree_nodes)
    plot!(plt_x_und, t_steps, history_x_und[:, node], label="Nodo $node", 
          color=colors[idx], marker=:circle)
    plot!(plt_p_und, t_steps, history_p_und[:, node], label="Nodo $node", 
          color=colors[idx], marker=:circle)
end

# Visualizziamo i plot
p_combined_und = plot(plt_x_und, plt_p_und, layout=(1, 2), size=(1200, 400))
display(p_combined_und)

# +
# ==================================================================
# SIMULAZIONE CON GRAFO FULLY CONNECTED (INDIRETTO)
# ==================================================================
println("\n" * "="^60)
println("SIMULAZIONE CON GRAFO FULLY CONNECTED - T=10, N=10")
println("="^60)

# Parametri (stessi di prima)
T = 50                    # numero di macro-step temporali
N = 10
a = 10.0
b = 2.0
c = 1.0
beta = 50.0
kappa = 0.5
s = fill(0.3, N)
T=50

# ==================================================================
# Costruzione di un grafo FULLY CONNECTED (completo)
# ==================================================================
# Ogni nodo è connesso a tutti gli altri con peso uniforme
G_fullconn = ones(Float64, N, N)
G_fullconn[diagind(G_fullconn)] .= 0.0  # Rimuovo auto-loop
G_fullconn .*= 0.3  # Peso uniforme su tutti gli archi

println("\nMatrice di adiacenza (fully connected):")
println("Ogni nodo connesso a tutti gli altri (tranne se stesso)")
println("Peso uniforme: 0.3")

# Visualizziamo il grafo fully connected
g_fullconn = Graph(G_fullconn .> 0)
gr()
plt_fullconn = graphplot(g_fullconn, names=1:N, nodesize=0.3, method=:spring,
                         markercolor=:lightgrey, curves=false)
display(plt_fullconn)

degrees_fc = degree(g_fullconn)
println("\nGradi dei nodi (tutti uguali in fully connected):")
println("Grado di ogni nodo: $(degrees_fc[1])")

# ==================================================================
# EVOLUZIONE
# ==================================================================
# Inizializziamo adozioni e prezzi
x_fc = rand(N)
p_fc = fill(3.0, N)

# Matrice storia
history_x_fc = zeros(T, N)
history_p_fc = zeros(T, N)

# Ciclo di evoluzione: 10 macro-step
for t in 1:T
    println("\n=== Macro step $t ===")
    println("stato iniziale x: ", round.(x_fc, digits=3))
    println("stato iniziale p: ", round.(p_fc, digits=3))

    # Monopolista calcola variazione di prezzo
    delta_p = exact_greedy_bellman(x_fc, p_fc, G_fullconn, a, b, c, s, kappa)
    p_fc .+= delta_p

    # Micro-aggiornamenti degli utenti
    simulate_users!(x_fc, p_fc, G_fullconn, a, b, beta, s, 3 * N)

    println("stato finale x: ", round.(x_fc, digits=3))
    println("stato finale p: ", round.(p_fc, digits=3))

    history_x_fc[t, :] = x_fc
    history_p_fc[t, :] = p_fc
end

# ==================================================================
# PLOT: Evoluzione per i 3 nodi con grado più alto
# ==================================================================
# In fully connected tutti hanno lo stesso grado, quindi ne selezioniamo 3 a caso
top_nodes_fc = [1, 4, 7]

println("\n=== Nodi selezionati (fully connected) ===")
for node in top_nodes_fc
    println("Nodo $node: grado=$(degrees_fc[node])")
end

# Plot temporali
t_steps = 1:T
plt_x_fc = plot(legend=:topright, xlabel="Macro-step", ylabel="Adozione x", 
                title="Evoluzione adozioni (grafo fully connected)", lw=2)
plt_p_fc = plot(legend=:topright, xlabel="Macro-step", ylabel="Prezzo p", 
                title="Evoluzione prezzi (grafo fully connected)", lw=2)

colors = [:red, :blue, :green]
for (idx, node) in enumerate(top_nodes_fc)
    plot!(plt_x_fc, t_steps, history_x_fc[:, node], label="Nodo $node", 
          color=colors[idx], marker=:circle)
    plot!(plt_p_fc, t_steps, history_p_fc[:, node], label="Nodo $node", 
          color=colors[idx], marker=:circle)
end

# Visualizziamo i plot
p_combined_fc = plot(plt_x_fc, plt_p_fc, layout=(1, 2), size=(1200, 400))
display(p_combined_fc)

# ==================================================================
# VERIFICA TEORICA: Prezzo statico per fully connected
# ==================================================================
println("\n" * "-"^60)
println("VERIFICA TEORICA: Grafo fully connected è simmetrico")
println("-"^60)

p_static_fc = fill((a + c) / 2, N)
p_final_fc = history_p_fc[T, :]
diff_fc = p_final_fc .- p_static_fc

println("Prezzo ottimale TEORICO statico: p* = $(a + c)/2 = $((a+c)/2)")
println("Prezzo FINALE dalla dinamica: ", round.(p_final_fc, digits=4))
println("Differenza media: ", round(mean(abs.(diff_fc)), digits=4))
println("\nNota: Anche fully connected dovrebbe convergere a p* = $((a+c)/2)")
println("      ma con dinamica greedy su $T step non ci arriva ancora.")
# -

# ## Greedy Bellman Equation 
#
# In this section we aim to solve computationally the greedy Bellman Equation with a finite time horizon $H$, shorter than the time horizon of the original problem $T$, to make it computationnaly tractable. For this we consider a synchronous dynamics where agents at each step set $t$ update their action as
# $$x_{i,t} = s_i x_{i,t-1} + (1-s_i) x_{i,t}^{NE}$$
# where $s_i$ is the intrinsic resistance of agent $i$ and $x_{i,t}^{NE} = \frac{a-p_{i,t}+ \sum_j g_{ij} x_{j,t-1}}{b}$ is the Nash Equilibrium level of consumption in which agent $i$ consider the rest of the network still frozen at the previous time step.  
# Because the dynamics is deterministic, the monopolist can simulate the system and solve the greedy Bellman Equation to choose the best price variation. The instantaneous reward for the monopolist is
# $$ r_t = 

# +
using LinearAlgebra
using Optim

# ==================================================================
# SOLUZIONE ESATTA BELLMAN H > 1 CON AZIONI CONTINUE
# ==================================================================

# 1. Funzione Obiettivo: calcola il profitto totale scontato per H passi
function H_step_profit(delta_P_flat, x_init, p_init, G, a, b, c, s, kappa, H, gamma)
    N = length(x_init)
    
    # L'ottimizzatore lavora con vettori "piatti". Lo rimodelliamo in una matrice [N utenti x H passi]
    delta_P = reshape(delta_P_flat, N, H) 
    
    x_curr = copy(x_init)
    p_curr = copy(p_init)
    total_profit = 0.0
    
    for t in 1:H
        dp = delta_P[:, t] # Azioni per il turno t
        
        # Dinamica Mean-Field (Deterministica)
        x_NE = (a .- (p_curr .+ dp) .+ G * x_curr) ./ b
        
        x_next = s .* x_curr .+ (1.0 .- s) .* x_NE
        x_next .= max.(0.0, x_next) # Le adozioni non possono essere negative
        p_next = p_curr .+ dp
        
        # Calcolo del profitto
        reward = sum((p_next .- c) .* x_next .- kappa .* (dp.^2))
        total_profit += (gamma^(t-1)) * reward
        
        # Avanzamento dello stato per il turno successivo
        x_curr .= x_next
        p_curr .= p_next
    end
    
    # Restituiamo il profitto col segno meno, perché la libreria Optim MINIMIZZA di default.
    # Minimizzare -Profitto equivale a Massimizzare il Profitto.
    return -total_profit 
end

# 2. Il "Cervello" del Monopolista
function exact_bellman_continuous(x_init, p_init, G, a, b, c, s, kappa, H; gamma=0.95)
    N = length(x_init)
    
    # Initial guess: partiamo con l'ipotesi di non cambiare nessun prezzo (tutti zeri)
    # Dimensioni: N utenti * H passi di tempo
    initial_guess = zeros(N * H)
    
    # Creiamo una funzione anonima (closure) che dipenda SOLO da delta_P_flat
    # (è il formato che richiede la libreria Optim)
    objective_function(dp) = H_step_profit(dp, x_init, p_init, G, a, b, c, s, kappa, H, gamma)
    
    # Lanciamo l'algoritmo di ottimizzazione non vincolata (L-BFGS è perfetto per funzioni quadratiche)
    result = optimize(objective_function, initial_guess, LBFGS())
    
    # Estraiamo la sequenza ottimale trovata
    optimal_sequence_flat = Optim.minimizer(result)
    optimal_sequence = reshape(optimal_sequence_flat, N, H)
    
    max_profit = -Optim.minimum(result)
    first_optimal_action = optimal_sequence[:, 1]
    
    return max_profit, first_optimal_action
end

# ==================================================================
# MAIN: ESECUZIONE 
# ==================================================================
N_users = 5
a_val = 10.0
b_val = 2.0
c_val = 1.0
beta_val = 5.0
kappa_val = 0.5
s_val = fill(0.3, N_users)

G_matrix = rand(N_users, N_users) .* 0.2
G_matrix[diagind(G_matrix)] .= 0.0 

x_init = rand(N_users) .* 2.0
p_init = fill(3.0, N_users)

println("--- Stato Iniziale ---")
println("Adozioni x: ", round.(x_init, digits=2))
println("Prezzi   p: ", round.(p_init, digits=2))
println("----------------------\n")


for orizzonte in [1, 2, 5, 10]
    tempo = @elapsed best_val, mossa_ottima = exact_bellman_continuous(
        x_init, p_init, G_matrix, a_val, b_val, c_val, s_val, kappa_val, orizzonte
    )
    
    println(">>> Risoluzione Esatta Continua per H = $orizzonte")
    println("Tempo di calcolo: ", round(tempo, digits=4), " secondi")
    println("Valore Atteso (Q): ", round(best_val, digits=2))
    println("Mossa Ottimale Δp (solo il primo step): ", round.(mossa_ottima, digits=3))
    println("")
end

# +
using Plots
using Plots.Measures

# ==================================================================
# 1. FUNZIONE WRAPPER PER LA SIMULAZIONE
# ==================================================================
# Inseriamo il ciclo in una funzione così possiamo chiamarlo facilmente per H diversi
function run_mpc_simulation(x_start, p_start, G, a, b, c, beta, s, kappa, H, T_sim)
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

# ==================================================================
# 2. ESECUZIONE DELLE DUE SIMULAZIONI (Miope vs Lungimirante)
# ==================================================================
T_sim = 40 # 40 istanti di tempo sono perfetti per vedere la convergenza

println("Avvio simulazione Miope (H = 1)...")
x_hist_H1, p_hist_H1 = run_mpc_simulation(
    x_init, p_init, G_matrix, a_val, b_val, c_val, beta_val, s_val, kappa_val, 1, T_sim
)

println("Avvio simulazione Lungimirante (H = 5)...")
x_hist_H5, p_hist_H5 = run_mpc_simulation(
    x_init, p_init, G_matrix, a_val, b_val, c_val, beta_val, s_val, kappa_val, 5, T_sim
)
println("Simulazioni completate. Generazione grafici...")

# ==================================================================
# 3. PLOTTING COMPARATIVO
# ==================================================================
nodes_to_plot = [1, 2, 3]
time_steps = 1:(T_sim + 1)
plot_size = (800, 450) 
margin_sinistro = 15mm

for i in nodes_to_plot
    # --- Grafico Adozione (x) ---
    # Linea tratteggiata azzurra per H=1, linea continua blu scuro per H=5
    plt_x = plot(time_steps, x_hist_H1[i, :], 
        label="H=1 (Miope)", linewidth=2, color=:cornflowerblue, style=:dash,
        title="Confronto Adozione (x) - Nodo $i", xlabel="Tempo", ylabel="Adozione x", 
        size=plot_size, left_margin=margin_sinistro, legend=:bottomright)
        
    plot!(plt_x, time_steps, x_hist_H5[i, :], 
        label="H=5 (Lungimirante)", linewidth=3, color=:darkblue)
        
    display(plt_x)

    # --- Grafico Prezzo (p) ---
    # Linea tratteggiata arancione per H=1, linea continua rossa per H=5
    plt_p = plot(time_steps, p_hist_H1[i, :], 
        label="H=1 (Miope)", linewidth=2, color=:orange, style=:dash,
        title="Confronto Prezzo (p) - Nodo $i", xlabel="Tempo", ylabel="Prezzo p", 
        size=plot_size, left_margin=margin_sinistro, legend=:topright)
        
    plot!(plt_p, time_steps, p_hist_H5[i, :], 
        label="H=5 (Lungimirante)", linewidth=3, color=:red)
        
    display(plt_p)
end
# -

# ## <center>Reinforcement Learning: SARSA</center>
#
# ### 1. Discrete State-Action Space
#
# To use reinforcement learning, we need discrete state-action space, but in our problem both consumption and prices are continuous. To implement *SARSA* algorithm, we discretize this state-action space in bins according to the parameters of the problem and the structure of the graph, assuming always Noisy Best Response for consumers.
#
# Let $x_i \in \mathbb{R}^+$ be the continuous state of user $i$ and the price variation $\Delta p_i \in \mathbb{R}$ the continuous action. We map these continuous variables into finite discrete sets $\mathcal{S}$ and $\mathcal{A}$.
#
# To prevent rapid changes in price, we bound the maximum price variation $\Delta p_{max}$ between $c$, the production cost, and $a$, that is the intrinsic utility parameter. 
# We define a discrete set of $K_a$ uniformly spaced actions
# $$\mathcal{A} = \{a_1, a_2, \dots, a_{K_a}\}$$
# where $a_1 = -\Delta p_{max}$ and $a_{K_a} = +\Delta p_{max}$. 
#
# To bound the consuption level of the agents, we consider the Nash equilibrium consumption of agent $i$
# $$ x_i^* = \frac{a - p_i}{b} + \frac{1}{b} \sum_j g_{ij} x_j $$
# Because the Noisy Best Response is equivalent to a Gaussian distribution, at $99.7\%$ the consumption is bounded by
# $$x_i \le \mu_i + 3\sigma = \mu_i + \frac{3}{\sqrt{\beta b}}$$
# We define $g_{max} = \max_i \sum_j g_{ij}$ be the maximum weighted in-degree of the network. To maximize $\mu_i$, we consider the minimum price $p_i=c$ and maximum positive externality $g_{max} x_{max} $
# $$\mu_{max} \le \frac{a - c}{b} + \frac{g_{max}}{b} x_{max}$$
# So the upper bound for $x_i$ is
# $$x_{max} \le \frac{a - c}{b} + \frac{g_{max}}{b} x_{max} + \frac{3}{\sqrt{\beta b}} \\
# x_{max} \le \frac{\frac{a - c}{b} + \frac{3}{\sqrt{\beta b}}}{1 - \frac{g_{max}}{b}}$$
#
# We then partition the interval $[0, x_{max}]$ into $K_s$ equally divided into bins and the continuous state $x_i$ is mapped to a discrete state index $S_i \in \{1, \dots, K_s\}$.
#
# ### 2. Reward Function
#
# The objective of the monopolist at time $t$ for user $i$ is the maximization of the profit. The  reward $r_{i,t}$ is 
# $$r_{i,t} = (p_{i,t} - c)x_{i,t} - \kappa (\Delta p_{i,t})^2$$
#
# ### 3. SARSA Learning Algorithm
#
# To learn the optimal pricing strategy, the monopolist employs *SARSA*. For each user $i$, the monopolist uses a Q-value function (discretized) $Q_i(S, A)$ representing the expected discounted future return of taking action $A$ in state $S$ and following the current policy.
#
# Action selection is governed by an $\epsilon$-greedy policy to balance exploration and exploitation
# $$
# A_t = 
# \begin{cases} 
# \text{random action} \in \mathcal{A} & \text{with probability } \epsilon \\
# \arg\max_{a \in \mathcal{A}} Q_i(S_t, a) & \text{with probability } 1 - \epsilon 
# \end{cases}
# $$
#
# At each macro-time step, after the network has asynchronously updated its adoptions through the Noisy Best Response dynamics, the monopolist observes the new discrete state $S_{t+1}$ and selects the next action $A_{t+1}$ using the $\epsilon$-greedy policy.
# The Q-value for the visited state-action pair is updated as follows
# $$Q_i(S_t, A_t) \leftarrow Q_i(S_t, A_t) + \alpha \left[ r_{i,t} + \delta Q_i(S_{t+1}, A_{t+1}) - Q_i(S_t, A_t) \right]$$
# where
# * $\alpha \in (0, 1]$ is the learning rate.
# * $\delta \in [0, 1)$ is the discount factor for future rewards.
#

# +
using LinearAlgebra
using Distributions
using StatsBase
using Plots
using Graphs

# ==================================================================
# 1. FUNZIONE DI SUPPORTO: MAPPATURA DELLO STATO
# ==================================================================
function get_state_idx(x_val, x_max, K_s)
    bin_width = x_max / K_s
    idx = ceil(Int, x_val / bin_width)
    return clamp(idx, 1, K_s)
end

# ==================================================================
# 2. DINAMICA DEGLI UTENTI: ASINCRONA CON ESPONENZIALE
# ==================================================================
function simulate_users!(x, p, G, a, b, beta, s, steps)
    N = length(x)
    x_candidati = 0.0:0.05:20.0 
    
    for _ in 1:steps
        i = rand(1:N) 
        
        if rand() > s[i] 
            esternalita = dot(G[i, :], x)
            U_candidati = [a * v - (b / 2.0) * v^2 + esternalita * v - p[i] * v for v in x_candidati]
            
            max_U = maximum(U_candidati)
            pesi = exp.(beta .* (U_candidati .- max_U))
            
            x[i] = sample(x_candidati, Weights(pesi))
        end
    end
end

# ==================================================================
# 3. ALGORITMO SARSA (Costante Epsilon-Greedy)
# ==================================================================
function train_sarsa!(x, p, G, a, b, c, beta, s, kappa, actions, x_max, K_s; 
                      episodes=500, steps_per_episode=50, alpha=0.1, gamma=0.95, epsilon=0.1)
    
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
                
                Q[i, S[i], A[i]] = Q_old + alpha * (reward + gamma * Q_next - Q_old)
            end
            
            S .= S_next
            A .= A_next
        end
    end
    
    return Q, x_history, p_history
end

# ==================================================================
# MAIN: SETUP E ESECUZIONE
# ==================================================================
N_users = 5
a_val = 10.0
b_val = 2.0
c_val = 1.0
beta_val = 5.0
kappa_val = 0.5
s_val = fill(0.3, N_users)

G_matrix = rand(N_users, N_users) .* 0.2
G_matrix[diagind(G_matrix)] .= 0.0 

x_init = rand(N_users) .* 2.0
p_init = fill(3.0, N_users)

K_s = 5
K_a = 5

g_max = maximum(sum(G_matrix, dims=2))
x_max = ((a_val - c_val) / b_val) * (1.0 + g_max) + (3.0 / sqrt(beta_val * b_val))

delta_p_max = (a_val - c_val) * 0.10
actions = collect(range(-delta_p_max, delta_p_max, length=K_a))

println("Limite x_max calcolato: ", round(x_max, digits=2))
println("Azioni Δp calcolate:    ", round.(actions, digits=3))

println("\nAvvio addestramento SARSA (Epsilon fisso a 0.1)...")
Q_learned, x_history, p_history = train_sarsa!(x_init, p_init, G_matrix, a_val, b_val, c_val, beta_val, s_val, kappa_val, actions, x_max, K_s, epsilon=0.1)

println("Addestramento completato.")
# -

# ## <center>ALGORITMO DI MATTEO PER IL 4 </center>

# +
using LinearAlgebra, Plots, Graphs, SparseArrays, Printf

N = 12
a, b, c = 26.0, 12.0, 4.0 
α, η, T = 0.5, 0.05, 1200
diretto = false  

K = 10

g = erdos_renyi(N, 0.3, is_directed=diretto)
G_raw = Float64.(adjacency_matrix(g))
max_sum = maximum(sum(G_raw, dims=2))
G = sparse(G_raw .* (b * 0.8 / max(1.0, max_sum)))

x = fill(0.1, N)
p = fill(a/2, N)    
history_p = zeros(T, N)

for t in 1:T
    x_BR = (fill(a, N) - p + G * x) ./ b
    x .= x .+ α .* (x_BR .- x)
    
    diff_p = p .- c
    z_approx = zeros(N)
    
    current_term = copy(diff_p) ./ b
    z_approx .+= current_term
    for k in 1:K
        current_term = (G' * current_term) ./ b
        z_approx .+= current_term
    end
    
    grad = x .- z_approx

    p .+= η .* grad
    
    history_p[t, :] .= p
end

# --- Analisi Finale ---
bonacich = inv(Matrix(I - (1/b) .* G')) * ones(N)
out_deg = outdegree(g)
perm = sortperm(p)

# Visualizzazione Risultati
header = @sprintf("Nodo | Out-D | Bonacich | Prezzo (K=%d)", K)
table_rows = [@sprintf("%2d   | %4d  | %8.3f | %7.3f", 
              perm[i], out_deg[perm[i]], bonacich[perm[i]], p[perm[i]]) for i in 1:N]
table_text = header * "\n" * "-"^45 * "\n" * join(table_rows, "\n")

p1 = plot(history_p, title="Prezzi con Dinamica Best-Response Locale", palette=:tab20, lw=2, legend=false)
p2 = plot(title="Stato Finale", grid=false, showaxis=false, ticks=false)
annotate!(p2, 0.1, 0.5, text(table_text, :left, 8, "Courier"))

display(plot(p1, p2, layout=(2,1), size=(800, 850)))
println(table_text)
