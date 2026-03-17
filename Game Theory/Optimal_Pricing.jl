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

# **Authors:** Emanuele Barbera, Marco Ghirardo, Matteo Grandinetti, Martino Pasqualotto  
# **Date:** March 2026  
#
# # <center>Optimal Pricing on Network</center>

#
#
# ## Introduction
#
# A monopolist sells a divisible good to $N$ consumers. The consumers are connected via a network $G=(V,E)$ defined by an adjacency matrix $G=(g_{ij})$, $i,j=1,\dots,N$. Each consumer $i$’s usage level $x_i$ depends on the goods of her neighbors through a positive externality.  
# Consumer $i$’s utility is linear-quadratic:
#
# $$u_i(x_i,x_{-i}) = ax_i - \frac{b}{2}x_i^2 + \sum_j g_{ij}x_ix_j - p_ix_i$$
#
# where $x_i$ is the *usage* of consumer $i$, $g_{ij}$ is the *weight of influence* from $j$ to $i$, $p_i$ is the price offered to $i$. Also $a>0$ is the *intrinsic utility parameter*, and $b>0$ is the *sensitivity to own consumption*. Consumers simultaneously choose usages to maximize their utilities given the price vector $p=(p_1,\dots,p_N)$. The monopolist sets $p$ in order to maximize the revenue:
#
# $$R(p) = \sum_i (p_i - c)x_i$$
#
# where $c \geq 0$ is the marginal cost of producing the good. We assume $a>c$ to ensure positive demand in isolation, and impose $b > \sum_j g_{ij} \ \forall i$ to guarantee a unique, stable equilibrium.
#
# ## <center>Consumers' Best Response and Nash Equilibrium</center>
#
# Consumer's utility function is $u_i(x_i,x_{-i}) = ax_i - \frac{b}{2}x_i^2 + \sum_j g_{ij}x_ix_j - p_ix_i$, that is concave in $x_i$, so to find the Nash equilibrium it's sufficient a first-order condition (assuming that the action space of each consumer is compact, and this is reasonable because it's bounded by $0$ and the total amount of the produced good):
#
# $$\begin{aligned}
# &\frac{\partial u_i}{\partial x_i} = (a - p_i) - b x_i + \sum_j g_{ij} x_j = 0 \\
# &\mathcal{B}_i(x_{-i}) \doteq x_i^* = \frac{a - p_i}{b} + \frac{1}{b} \sum_j g_{ij} x_j
# \end{aligned}$$
#
# Thus, in vectorial form, the consumers' usage Nash Equilibrium is:
#
# $$\begin{aligned}
# x^* &= \frac{a𝟙 - p}{b} + \frac{1}{b} G x^* \\
# x^* &= \left(I - \frac{1}{b}G\right)^{-1} \frac{a𝟙 - p}{b}
# \end{aligned}$$
#
# where $𝟙$ is a $N$-dimensional vector of all $1$. 
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
# So, if $b > \sum_j g_{ij} \ \forall i$, we have that $\rho(G) < 1$ and $(I - \frac{1}{b}G)^{-1} = \sum_{i=0}^\infty \left(\frac{1}{b}G\right)^i$.
#
# ## <center>Optimal Pricing</center>
#
# If agents consume according to the Nash equilibrium, defining $M = (I - \frac{1}{b}G)^{-1}$, the revenue for the monopolist is
#
# $$\begin{aligned}
# R(p) &= (p - c𝟙)^\top x^* = (p - c𝟙)^\top M \frac{a𝟙 - p}{b}\\
# &= \frac{1}{b} \left[ -p^\top M p + p^\top M 𝟙 a + 𝟙^\top M p c - 𝟙^\top M 𝟙 ac \right]
# \end{aligned}$$
#
# To maximize $R(p)$ it's sufficient to impose $\nabla_p R = 0$:
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
# where $\beta \in [0,1]$ and $\mu \in \mathbb{R}^N$ is the intrinsic centrality of each node.\
# In the expression of the optimal price we can interpret the term $c𝟙$ as a basis price for everyone, $(a-c)𝟙$ as an intrinsic centrality $\mu$ proportional to the difference between the intrinsic utility parameter $a$ and the marginal cost $c$, and $M = (I-\frac{G}{b})^{-1}$ as measure of how much the agents are influenced by other agents in the network. We can conclude that the agents that are more influenced by others will get higher prices, while agents that have a strong influence will have lower prices. We'll deepen this consideration studying the problem on different networks in the next section.
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
#
# *From now on, unless noted otherwise, we'll refer to $x^*$ as the consumers' usage Nash Equilibrium: that is to interpret as the Nash equilibrium corresponding to the optimal price.*
#
# We have implemented the necessary functions to simulate these results in the *static_analysis* and *plot_functions* file: we are widely using them for the simulations in the following section.

using Plots, Graphs, LinearAlgebra, GraphRecipes, SparseArrays
include("static_analysis.jl")
include("plot_functions.jl")

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
# Instead, the consumers' usage Nash Equilibrium is:
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
#number of agents
N = 9   

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = createGraphFullyConnected(N)

#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)


#theoretical results
x_star_explicit = (0.5 * (a - c) / (b - (N - 1))) * ones(N)
p_star_explicit = 0.5 * (a + c) * ones(N)


println("x_star_th: ", round.(x_star_explicit, digits=4))
println("p_star_th: ", round.(p_star_explicit, digits=4))
# -

plot2graphs(G, p_star, x_star)

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
# The consumers' usage Nash Equilibrium
#
# $$x^* = M\frac{a𝟙-p^*}{b} = \frac{1}{2}\frac{a-c}{b}M𝟙 = \frac{1}{2}\frac{a-c}{b}\frac{b}{b-1}𝟙 = \frac{1}{2}\frac{a-c}{b-1}𝟙$$
#
# is analogous to the one obtained in the fully connected graph, considering that in this case $\sum_jg_{ij} = 1 \ \forall i$. Considerations similar to the ones for the fully connected graph can be done.

# +
#number of agents
N = 9   

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = createDirectedGraphRing(N)

#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)

#theoretical results
x_star_explicit = (0.5 * (a - c) / (b - 1)) * ones(N)
p_star_explicit = 0.5 * (a + c) * ones(N)

println("x_star th: ", round.(x_star_explicit, digits=4))
println("p_star th: ", round.(p_star_explicit, digits=4))
# -

plot2graphs(G, p_star, x_star)

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
#number of agents
N = 9   

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = adjacency_matrix(random_regular_graph(N, d))
G = Matrix(G) * 1.0

#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)

#theoretical results
w = sum(G[1, :])
x_star_explicit = (0.5 * (a - c) / (b - w)) * ones(N)
p_star_explicit = 0.5 * (a + c) * ones(N)

println("x_star th: ", round.(x_star_explicit, digits=4))
println("p_star th: ", round.(p_star_explicit, digits=4))


# -

plot2graphs(G, p_star, x_star)

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
#number of agents
N = 9   

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = createUndirectedGraphStar(N)

#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)

#theoretical results
z1 = b * (b + N - 1) / (b^2 - (N - 1))
zk = b * (b + 1) / (b^2 - (N - 1))
z = [z1; fill(zk, N-1)]
x_star_explicit = (0.5 * (a - c) / b) * z
p_star_explicit = 0.5 * (a + c) * ones(N)

println("x_star th: ", round.(x_star_explicit, digits=4))
println("p_star th: ", round.(p_star_explicit, digits=4))


# -

plot2graphs(G, p_star, x_star, lay = :stress)

# ### Directed Star Graph: Single Influencer and Isolated Followers
#
# We recall that, by looking at the best response for each agent $ \mathcal{B}_i(x_{-i})= \frac{a-p_i}{b}+\frac{1}{b}\sum_jg_{ij}x_j$, we must interpret a directed edge $(i,j)$ from $i$ to $j$ as the influence that $j$ has on $i$. It follows that to model a star graph with a central influencer we must construct a graph in which all the leaves in the star have an outgoing edge connecting them to the central hub. Namely:
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
# $\delta^{(1)}$ being the vector of all zeros apart from a $1$ in the first position.
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
#number of agents
N = 9   

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = createDirectedGraphStar(N)

#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)

#theoretical results
denom = 4 * b^2 - (N - 1)
phi1 = (2 * b^2 - (b + 1) * (N - 1)) / denom
phil = (2 * b^2 + b) / denom
phi = [phi1; fill(phil, N-1)]

x_explicit = (a - c) / b^2 * [b * (1 - phi1); fill((1 - phi1) + b * (1 - phil), N-1)]
p_star_explicit = c * ones(N) + (a - c) * phi

println("x_star th: ", round.(x_explicit, digits=4))
println("p_star th: ", round.(p_star_explicit, digits=4))
# -

plot2graphs(G, p_star, x_star, lay = :stress)

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
#    p^*_{\text{connected follower}} - p^*_{\text{isolated follower}}   = (a-c)\left[ \frac{(2b+1)(b-(N-2))}{4b^2 - 4b(N-2) - (N-1)} -\frac{2b^2+b}{4b^2-(N-1)} \right] > 0 \ \ \  \forall b, N,a,c |\  b >N-1, N>2, a>c
# \end{align*}
# That is to say the positive externality added in this second case (if we're in a well posed problem with $N>2$) causes the  price on the followers to rise.\
# On the other hand:
# \begin{align*}
#    p^*_{\text {hub with connected followers}} - p^*_{\text{hub with isolated followers}}   = (a-c) \left[\frac{2b^2 - b(3N-5) - (N-1)}{4b^2 - 4b(N-2) - (N-1)}-\frac{2b^2-(b+1)(N-1)}{4b^2-(N-1)}\right]  < 0 \ \ \  \forall b, N,a,c |\  b >N-1, N>2, a>c
# \end{align*}
# The connectedness of followers causes the price on the influencer to diminish. 
#
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
# $x^*_1 < x^*_L$. This means that not only the central hub gets a price lower than the price for the leaves and lower than the one it got in the case of isolated followers: it is also pushed to consume less than its followers.

# +
#number of agents
N = 9   

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = zeros(N, N)
G[2:N, 2:N] = createGraphFullyConnected(N-1)
G[2:N, 1] .= 1.0


#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)


#theoretical results
denom = 4*b^2-4*b*(N-2)-(N-1)
phi1 = (2*b^2-b*(3*N-5)-(N-1))/ denom
phil = ((2*b+1)*(b-(N-2))) / denom
phi = [phi1; fill(phil, N-1)]


x_star_explicit = (a - c) / b * [(1 - phi1); fill((b)/(b-(N-2))*((1 - phi1)/b + 1 - phil), N-1)]


p_star_explicit = c * ones(N) + (a - c) * phi
println("x_star th: ", round.(x_star_explicit, digits=4))
println("p_star th: ", round.(p_star_explicit, digits=4))



denom = 4 * b^2 - (N - 1)
phil_isolated = (2 * b^2 + b) / denom
phi1_isolated = (2 * b^2 - (b + 1) * (N - 1)) / denom
p_star_isolated_l = c + (a - c) * phil_isolated
p_star_isolated_1 = c + (a - c) * phi1_isolated


pl_diff = p_star[2] - p_star_isolated_l
p_hub_diff = p_star[1] - p_star_isolated_1

println("Prezzo_connected_follower - Prezzo_isolated_follower: ", round(pl_diff, digits=4))
println("Prezzo_hub_connected_flws - Prezzo_hub_isolated_flws: ", round(p_hub_diff, digits=4))
# -

plot2graphs(G, p_star, x_star, lay = :shell)

# ### Multiple Isolated Influencers and Communities of Followers
#
# We construct a graph with $N_{inf}$ isolated influencers and the remaining nodes organized in communities of $S_{com}$ nodes each.
#
#    

# +
#number of influencers
N_infl = 1
#number of communities
n_com = 2
#size of the communities
S_com = 4

#number of agents
N = N_infl + n_com*S_com

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = createGraphInfluencersCommunities(N_infl, S_com, n_com)


#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)
# -

plot2graphs(G, p_star, x_star, lay = :shell)

# Può essere utile confronare il caso con un influencer e due comunità da 4 vs il caso di prima con 1 influencer e 8 fully connected

# INFLUENCER DEGLI INFLUENCERS E COMUNITà

# +
#number of 2nd-level influencers
N_infl = 3
#number of communities
n_com = 3;
#size of the communities
S_com = 2

#number of agents
N = N_infl + n_com*S_com + 1

#parameters
a = 10.0
b = 12.0
c = 2.0

#adjacency matrix
G = zeros(N, N)
G[2:N_infl+1, 1] .= 1.0
G[2:N, 2:N] = createGraphInfluencersCommunities(N_infl, S_com, n_com)
G[2:N_infl+1, 2: N_infl+1] =createGraphFullyConnected(N_infl)


#analyse graph and find x_star and p_star
x_star, p_star = static_graph_analysis(G, a,b,c)
# -

plot2graphs(G, p_star, x_star)

# ## <center>Critical cases<center>
#
# We've already computed the Nash Equilibrium value for the users (in this section, we will indicate $x^*$ as the Nash Equilibrium)
# $$x^* = \frac{a𝟙 - p}{b} + \frac{1}{b} G x^*$$
# that is, if $I - \frac{1}{b}G$ is invertible
# $$x^* = \left(I - \frac{1}{b}G\right)^{-1} \frac{a𝟙 - p}{b}$$
# We now analyze 2 different situations.
# ### First case: $I - \frac{1}{b}G$ not invertible
# We know that a matrix $A \in \mathbb{R}^{N\times N}$ is invertible if and only if $\det(A) \neq 0$. Since the eigenvalues of the matrix $\left(I - \frac{1}{b}G\right)$ are given by $1 - \frac{\lambda_i}{b}$ (where $\lambda_i$ are the eigenvalues of $G$), the determinant $\det\left(I - \frac{1}{b}G\right) = 0$ if and only if there exists an eigenvalue of $G$ exactly equal to $b$. In this case, depending on the value of $p$, the system
# $$(I -\frac{1}{b} G) x^* = \frac{a𝟙 - p}{b}$$
# could have $0$ or infinite solutions, according to the Fredholm theorem.  
# The Fredholm theorem states that a linear system $Ax=y$, where $A \in \mathbb{R}^{N \times N}$, $x,y \in \mathbb{R}^N$ and $\det(A)=0$, admits a solution if and only if the constant vector $y$ is orthogonal to the left eigenvector of the matrix $A$ corrisponding to the eigenvalue $0$.  
# If $x_1$ is a solution to the system $Ax=y$, all vector in the form
# $$ x_2 = x_1 + k v_1 $$
# are solutions to the system, where $k \in \mathbb{R}$ is a generic constant and $v_1$ is the right eigenvector of $A$ corresponding to the eigenvalue $0$.  
# We analyze the case where $G$ is regular, i.e. $G𝟙=w𝟙$, and we start from $x_0 = q 𝟙$, with $q>0$, and constant prices. By substituing in the system
# \begin{align*}
# (I -\frac{1}{b} G) q 𝟙 &= \frac{a - p}{b} 𝟙 \\
# (q - \frac{qw}{b}) 𝟙 &= \frac{a - p}{b} 𝟙 \\
# q &= \frac{a-p}{b-w}
# \end{align*} 
#
# We simulate the cases where the system has zero or infinite solutions.  
# We first try to solve the equation $x^* = \frac{a\mathbf{1} - p}{b} + \frac{1}{b} G x^*$ by iteration, starting from a uniform level of consumption. This corresponds to a Best Response Dynamics, where the agents iteratively update their actions as
# $$x_{i,t+1} = x_{i,t}^{BR} = \frac{a\mathbf{1} - p}{b} + \frac{1}{b} G x_{t}$$
# We will see that in this first scenario the dynamics diverges both in the cases in which we have zero or infinite solutions.  
# The only case in which this dynamics doesn't diverge is when $b=\rho(G)$ and the system $x^* = \frac{a\mathbf{1} - p}{b} + \frac{1}{b} G x^*$ admits solution. In this case the Best Response Dynamics converges, if $G$ is not bipartite, to a Nash Equilibrium out of the infinite ones.
#
# Then, we consider the case where, by starting with uniform prices and uniform consumption, the orthogonality condition is met and we obtain infinite solutions. Below, we plot the trajectories corresponding to the formula $x_2 = x_1 + k v_1$, where $x_1 = q 𝟙$, for the arbitrary constants $k \in \{-3, 0, 3\}$.

# +
using Plots, Graphs, GraphRecipes, LinearAlgebra

N = 5
a = 10.0  

G = [0.0  0.8  0.1  0.1  0.0;
     0.8  0.0  0.1  0.1  0.0;
     0.1  0.1  0.0  0.8  0.0;
     0.1  0.1  0.8  0.0  0.0;
     0.25 0.25 0.25 0.25 0.0]  # In this case G is a stochastic matrix, G𝟙 = 𝟙 (w = 1)
    
b = 0.6 # corresponds to the second largest eigenvalue of G

v = [1.0, 1.0, -1.0, -1.0, 0.0] # left and right eigenvector of I - G/b associated to the eigenvalue λ = 0

labels = reshape(["User $i" for i in 1:N], 1, N)
colors = [:blue, :cyan, :red, :orange, :green]
steps = 15

p_net = graphplot(G, names=1:N, nodesize=0.2, curves=false, markercolor=:lightgrey,
                  size=(600, 400))
display(p_net)

# CASE 1A: Best Response Dynamics - Zero Solutions

p_zero = [8.0, 8.0, 10.0, 10.0, 9.0]
x_zero_history = zeros(N, steps)
x_curr_zero = zeros(N) 

for t in 1:steps
    x_zero_history[:, t] = x_curr_zero
    x_curr_zero = (a .- p_zero) ./ b .+ (G * x_curr_zero) ./ b
end

p_div_zero = plot(title="Divergence (Zero Solutions)",
                  xlabel="iterations", ylabel="x", legend=:topleft)
for i in 1:N
    plot!(p_div_zero, 1:steps, x_zero_history[i, :], 
          lw=3, color=colors[i], label="User $i")
end

display(p_div_zero)

# CASE 1B: Best Response Dynamics - Infinite Solutions

p_inf = fill(12.0, N)
x_inf_brd_history = zeros(N, steps)
x_curr_inf_brd = fill(6.0, N)  

for t in 1:steps
    x_inf_brd_history[:, t] = x_curr_inf_brd
    x_curr_inf_brd = (a .- p_inf) ./ b .+ (G * x_curr_inf_brd) ./ b
end

p_div_inf = plot(title="Divergence (Infinite Solutions)",
                 xlabel="iterations", ylabel="x", legend=:bottomleft)
for i in 1:N
    plot!(p_div_inf, 1:steps, x_inf_brd_history[i, :], 
          lw=3, color=colors[i], label="User $i")
end

display(p_div_inf)

# CASE 2: Convergence to a NE when b = ρ(G) - Infinite Solutions

b_rho = 1.0 
p_rho = fill(10.0, N) 

x_rho_history = zeros(N, steps)
x_curr_rho = [4.0, 8.0, 7.0, 9.0, 5.0] 

for t in 1:steps
    x_rho_history[:, t] = x_curr_rho
    x_curr_rho = (a .- p_rho) ./ b_rho .+ (G * x_curr_rho) ./ b_rho
end

p_conv_rho = plot(title="Convergence to a NE (b = ρ(G))",
                  xlabel="iterations", ylabel="x", legend=:topleft)
for i in 1:N
    plot!(p_conv_rho, 1:steps, x_rho_history[i, :], 
          lw=3, color=colors[i], label="User $i")
end

display(p_conv_rho)

# CASE 3: Infinite Solutions (Stationary states)

w = 1.0
x_base = fill((a - p_inf[1]) / (b - w), N) 

graphs = []

for k in [-3.0, 0.0, 3.0]
    x_curr_inf = x_base .+ k .* v
    
    plt_inf = plot(title="Infinite solutions (k = $k)",
                   xlabel="t", ylabel="x", ylims=(0, 10), legend=:topleft)
                  
    for i in 1:N
        hline!(plt_inf, [x_curr_inf[i]], lw=3, color=colors[i], label="User $i")
    end
    
    push!(graphs, plt_inf)
end

final_graph = plot(graphs..., layout=(3, 1), size=(800, 900))
display(final_graph)
# -

# ### Second case: $\exists i$ such that $b < \sum_j g_{ij}$
# When the sum of the influences for at least one user $i$ exceeds $b$, the system loses its stability.  
# The spectral radius of the matrix $\frac{1}{b}G$ can become greater than $1$, so the matrix $\left(I - \frac{1}{b}G\right)$ might still be invertible, but the Nash Equilibrium is not stable.  
# The network can enter a situation where if a user increases their consumption slightly, their neighbors will increase theirs to match the positive externality. This can push $x^*$ towards infinity.  
# If the system has a solution and the spectral radius of the matrix $\frac{1}{b}G$ is greater than $1$, this solution is not stable, meaning that, if the system is in his Nash Equilibrium, a little perturbation can make the consumption of the agents diverge, and in this case it has no economic sense because consumption cannot be infinite.  
# This can be proven rigorously assuming a Best Response Dynamics for the agents.  
# The Best Response dynamics at time $t$ is given by $x_{t+1} = \frac{a𝟙 - p}{b} + \frac{1}{b}G x_t$. We define the vector $\varepsilon_t = x_t - x^*$ as the distance from the equilibrium. Subtracting the static equation from the dynamic one, we get
# $$\varepsilon_{t+1} = x_{t+1} - x^* = \frac{1}{b}G(x_t - x^*) = \frac{1}{b}G \varepsilon_t$$
# Iterating from $t=0$, the evolution of $\varepsilon_t$ is
# $$\varepsilon_t = \left(\frac{1}{b}G\right)^t \varepsilon_0$$
# If the spectral radius of $\frac{G}{b}$ is greater than $1$, $||\varepsilon_t||_\infty \to \infty$ as $t \to \infty$.
#
# Furthermore, if $a \ge p_i \ \forall i$, considering the system $(I -\frac{1}{b} G) x^* = \frac{a𝟙 - p}{b}$, the term $\frac{a𝟙 - p}{b}$ is non-negative. For Perron-Frobenius theorem, the left and right eigenvector of $\frac{G}{b}$ corresponding to its greatest eigenvalue, $\lambda_{max} = \rho(\frac{G}{b}) \doteq \rho$, are non-negative. Let $u$ be the left eigenvector of $\frac{G}{b}$ corresponding to its greatest eigenvalue, we consider the following expression
# \begin{align*}
# u^\top \cdot (I -\frac{1}{b} G) x^* &= u^\top \cdot \frac{a𝟙 - p}{b} \\
# u^\top \cdot  x^* (1-\rho) &= u^\top \cdot \frac{a𝟙 - p}{b} \\
# \end{align*}
# Because $u$ and $\frac{a𝟙 - p}{b} $ are non-negative, if $\rho >1$ we can conclude that at least one element of $x^*$ is negative, but that doesn't have a physiscal meaning.
#
# To guarantee a stable, finite, and positive Nash Equilibrium, the condition $b > \sum_j g_{ij}$ must hold for all users.
#
# We first simulate the case where $\rho(\frac{G}{b}) >1$, showing that, starting from $x=(0,\dots,0)$, with a Best Response Dynamics, the system diverges, and 
# we then simulate the case where $\rho(\frac{G}{b}) < 1$ and the Best Response Dynamics converges.  
# If $\rho(\frac{G}{b})=1$, this implies that exists an eigenvalue of $I - \frac{G}{b}$ is equal to $0$ and so $I - \frac{G}{b}$ is not invertible, a situation that we've already analyzed.

# +
N = 5
G = ones(N, N) - I
λ_max = maximum(eigvals(G))

a = 10.0
p = [2.0, 2.5, 3.0, 3.5, 4.0] 

steps = 15
labels = reshape(["User $i" for i in 1:N], 1, N)
colors = [:blue, :cyan, :red, :orange, :green]

graphs = []

# CASE 1: ρ > 1 

b_div = 3.0 
x_curr = zeros(N)
x_hist_div = zeros(N, steps)
x_star_div = inv(I - G ./ b_div) * ((a .- p) ./ b_div)

for t in 1:steps
    x_hist_div[:, t] = x_curr
    x_curr = (a .- p) ./ b_div .+ (G * x_curr) ./ b_div
end

y_min = minimum(x_star_div) - 2.0
y_max = maximum(x_hist_div) + 5.0

p1 = plot(title="1. Divergence (ρ > 1)", xlabel="iterations", ylabel="x",
          ylims=(y_min, y_max), legend=:outerright)

for i in 1:N
    plot!(p1, 1:steps, x_hist_div[i, :], lw=2, color=colors[i], label="User $i")
    hline!(p1, [x_star_div[i]], color=colors[i], lw=1, ls=:dash, label="NE $i: $(round(x_star_div[i], digits=1))")
end
push!(graphs, p1)

# CASE 2: ρ < 1

b_conv = 5.0
x_curr = zeros(N)
x_hist_conv = zeros(N, steps)
x_star_stable = inv(I - G ./ b_conv) * ((a .- p) ./ b_conv)

for t in 1:steps
    x_hist_conv[:, t] = x_curr
    x_curr = (a .- p) ./ b_conv .+ (G * x_curr) ./ b_conv
end

p2 = plot(title="2. Convergence (ρ < 1)", xlabel="iterations", ylabel="x",
          legend=:outerright)

for i in 1:N
    plot!(p2, 1:steps, x_hist_conv[i, :], lw=2, color=colors[i], label="User $i")
    hline!(p2, [x_star_stable[i]], color=colors[i], lw=1, ls=:dash, label="NE $i: $(round(x_star_stable[i], digits=1))")
end
push!(graphs, p2)

plot(graphs..., layout=(2,1), size=(900, 900), margin=5Plots.mm)
# -

# # <center>Continuous-time Dynamics and Optimal Control</center>
#
# Let's consider the adoption dynamics due to the following continuous-time relaxation towards best response:
#
# $$\begin{aligned}
# \dot{x}(t) &= -\Lambda \left(x(t) - \frac{1}{b}(a𝟙 - p(t) + Gx(t))\right) = \\
# &= -\Lambda\left( I - \frac{1}{b} G\right) x(t) + \frac{1}{b}\Lambda(a𝟙 - p(t)) \doteq \\
# &\doteq D x(t) + f(t)
# \end{aligned}$$
#
# where $\Lambda=\text{diag}(\pi_i)$ is an individual update rate we defined $D \doteq -\Lambda\left( I - \frac{1}{b} G\right)$ and $f(t) \doteq \frac{1}{b}\Lambda(a𝟙 - p(t))$. 
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
#     \dot{z} &= \begin{pmatrix} -\Lambda\left( I - \frac{1}{b} G\right) & -\frac{1}{b}\Lambda \\ 0 & 0 \end{pmatrix} z + \begin{pmatrix} 0 \\ I \end{pmatrix} u + \begin{pmatrix} \frac{a}{b}\Lambda 𝟙 \\ 0 \end{pmatrix} \doteq \\
#     &\doteq Az+Bu+d
# \end{aligned}$$
#
# where we defined $A \doteq \begin{pmatrix} -\Lambda(I - \frac{1}{b}G) & -\frac{1}{b}\Lambda\\ 0 & 0 \end{pmatrix}$, $B \doteq \begin{pmatrix} 0 \\ I \end{pmatrix}$ and $d \doteq \begin{pmatrix} \frac{a}{b}\Lambda 𝟙 \\ 0 \end{pmatrix}$. The intertemporal revenue becomes a quadratic form in the augmented space:
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
# The PMP provides necessary conditions for optimality. However, in this linear-quadratic problem, since the dynamics are linear, the objective function is concave in $u$ and the condition $b > \sum_j g_{ij} \ \forall i$ prevents the state variables $(x, p)$ from diverging, the first-order conditions derived from the PMP are also sufficient for a global maximum.
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
# \dot{x}(t) &= -\Lambda \left(x(t) + \frac{1}{b}(a𝟙 - p(t) + Gx(t))\right)\\
# u(t) &= \dot{p}(t) = -K^{-1} B^\top \left( S(t) z(t) + \nu(t) \right)
# \end{cases}$$

# ## Simulation

# +
include("static_analysis.jl")
include("continuous_control.jl")
include("plot_functions.jl")
  
"""PARAMETERS"""
N = 9
a = 10.0
b = 12.0
c = 2.0 

G = createGraphFullyConnected(N, directed=true)

x_star, p_star = static_graph_analysis(G, a, b, c)

"""DYNAMICAL PARAMETERS"""
𝛿 = 1.0             #time discount
Λ = 0.2*diagm(ones(N)+0.25*rand(N)-0.25*rand(N))

k = 1.0
K = k*diagm(ones(N))

T_0 = 0.0
T_F = 100.0
Δt = 0.01

times = Vector(T_0:Δt:T_F)
steps = length(times)

"""AUXILIARY MATRICES"""

A = [Λ * ( (G/b) - I )  -Λ/b;
     zeros(N, N) zeros(N, N)]

B = [zeros(N, N); I(N)]

d = vcat(Λ * (a/b) * ones(N), zeros(N));
# -

# We solve the first system integrating backwards for $S$ and $\nu$, then choosing some random initial conditions, we integrate forward in time.

# +
S, v= integrate_backward(A, B, d, K, N, 𝛿, Δt, steps);

# Initial condition near NE
x0 = x_star.*(ones(N)+0.1*rand(N)-0.1*rand(N))
p0 = p_star.*(ones(N)+0.1*rand(N)-0.1*rand(N))

println("\nINITIAL CONDITIONS")
println("x0: $(round.(x0, digits = 4))")
println("p0: $(round.(p0, digits = 4))")

X, P, U, x_star = integrate_forward(x0, p0, S, v, G, K, Λ, B, N, a, b, c, Δt, steps)

println("Final usages: $(round.(X[:,steps], digits = 4))")
println("Final prices: $(round.(P[:,steps], digits = 4))")
# -

plot_continuous_episode(X, P, times, T_0, T_F)

# +
my_rewards = calculate_reward(times, X, P, U, c, 𝛿, k)

plot_continuous_reward(my_rewards, times, T_0, T_F)
# -

# # <center>Discrete-time control</center>

# ## Greedy Bellman Equation with Asynchronous Noisy Best Response
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
# The monopolist aims to find the optimal price variation $\Delta p_i$ to maximize expected profit at the next macro-time step. For this, we use a greedy Bellman Equation with a time horizon $H=1$. The reward function for the monopolist includes a penalty $\kappa (\Delta p_i)^2$ to prevent rapid changes in prices.
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

# ## Simulation

# +
using LinearAlgebra
using Distributions
using Graphs, GraphRecipes, Plots

include("discrete_control.jl")
include("plot_functions.jl")
include("static_analysis.jl")

T = 50                    # numero di macro-step temporali

"""GRAPH INITIALIZATION"""
N = 8                    
a = 10.0
b = 12.0
c = 1.0

G = createGraphFullyConnected(N, directed=true)

static_graph_analyis(G,a,b,c)

"""DYNAMICAL PARAMETERS"""

beta = 1000.0
kappa = 0.5
s = fill(0.3, N)           # vettore resistenze uniforme

"""INITIAL CONDITIONS"""
x = rand(N)
p = fill(3.0, N)

states = zeros(N, T)
prices = zeros(N, T)

"""SIMULATION WITH GREEDY BELLMAN RESPONSE"""

for t in 1:T

    # evoluzione: T macro‑step, ciascuno con 3·N micro‑step
    p .+= exact_greedy_bellman(x, p, G, a, b, c, s, kappa)

    simulate_users!(x, p, G, a, b, beta, s, 3 * N)

    states[:,t] = x
    prices[:,t] = p
end

plot_discrete_episode(states, prices, 1, T)
# -

# ## Greedy Bellman Equation with sinchronous deterministic dynamics
#
# In this section we aim to solve computationally the greedy Bellman Equation with a finite time horizon $H$, shorter than the time horizon of the original problem $T$, to make it computationnaly tractable. For this we consider a synchronous dynamics where agents at each step set $t$ update their action as
# $$x_{i,t} = s_i x_{i,t-1} + (1-s_i) x_{i,t-1}^{NE}$$
# where $s_i$ is the intrinsic resistance of agent $i$ and $x_{i,t-1}^{NE} = \frac{a-p_{i,t}+ \sum_j g_{ij} x_{j,t-1}}{b}$ is the Nash Equilibrium level of consumption in which agent $i$ consider the rest of the network still frozen at the previous time step.  
# Because the dynamics is deterministic, the monopolist can simulate the system and solve the greedy Bellman Equation to choose the best price variation. We choose the instantaneous reward for the monopolist to be
# $$ r_t = (p_t-c) \cdot x_t - \kappa (p_t - p_{t-1})^2$$
# The first term is the net gain from the consumes, as in the first section, the second term introduces a cost for price variation, a resistance that was considered also in the continuous time section.
# We introduce a time discount for the total reward of the monopolist and we choose exponential time discounting to ensure that the Bellman principle of Optimality is valid and to have time consistency. So the total reward is
# $$ R = \sum_{t=1}^H \delta^{t-1} r_t$$
# To solve computationally the Bellman Equation we use the library *Optim*.

# +
using Optim

include("discrete_control.jl")

x_init = rand(N) .* 2.0
p_init = fill(3.0, N)

println("--- Stato Iniziale ---")
println("Adozioni x: ", round.(x_init, digits=2))
println("Prezzi   p: ", round.(p_init, digits=2))
println("----------------------\n")


for H in [1, 2, 5, 10]

    tempo = @elapsed best_val, mossa_ottima = exact_bellman_continuous(x_init, p_init, G, a, b, c, s, kappa, H)
    
    println(">>> Risoluzione Esatta Continua per H = $H")
    println("Tempo di calcolo: ", round(tempo, digits=4), " secondi")
    println("Valore Atteso (Q): ", round(best_val, digits=2))
    println("Mossa Ottimale Δp (solo il primo step): ", round.(mossa_ottima, digits=3))
    println("")
end

# +
using Plots
using Plots.Measures

T = 40

println("Avvio simulazione Miope (H = 1)...")
x_hist_H1, p_hist_H1 = run_simulation(
    x_init, p_init, G_matrix, a, b, c, beta, s_val, kappa, 1, T)

println("Avvio simulazione Lungimirante (H = 5)...")
x_hist_H5, p_hist_H5 = run_simulation(
    x_init, p_init, G_matrix, a, b, c, beta, s_val, kappa, 5, T)
println("Simulazioni completate. Generazione grafici...")

# ==================================================================
# PLOTTING COMPARATIVO
# ==================================================================
nodes_to_plot = [1, 2, 3]
time_steps = 1:(T + 1)
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
# To use reinforcement learning, we need discrete state-action space, but in our problem both consumption and prices are continuous. To implement *SARSA* algorithm, we discretize this state-action space in bins according to the parameters of the problem and the structure of the graph, assuming Noisy Best Response for consumers.
#
# Let $x_i \in \mathbb{R}^+$ be the continuous state of user $i$ and the price variation $\Delta p_i \in \mathbb{R}$ the continuous action. We map these continuous variables into finite discrete sets $\mathcal{S}$ and $\mathcal{A}$.
#
# To prevent rapid changes in price, we bound the maximum price variation $\Delta p_{max}$, so that the new $p$ is bounded between $c$, the production cost, and $a$, the intrinsic utility parameter. 
# We define a discrete set of $K_a$ uniformly spaced actions
# $$\mathcal{A} = \{a_1, a_2, \dots, a_{K_a}\}$$
# where $a_1 = -\Delta p_{max}$ and $a_{K_a} = +\Delta p_{max}$. 
#
# To bound the consumption level of the agents, we consider the Nash equilibrium consumption 
# $$ x_i^* = \frac{a - p_i}{b} + \frac{1}{b} \sum_j g_{ij} x_j^* $$
# Because in this case the Noisy Best Response is equivalent to a Gaussian distribution, at $99.7\%$ the consumption is bounded by
# $$x_i \le \mu_i + 3\sigma = \mu_i + \frac{3}{\sqrt{\beta b}}$$
# We define $g_{max} = \max_i \sum_j g_{ij}$ be the maximum weighted out-degree of the network. To find an upper bound for $\mu_i$, we consider the minimum price $p_i=c$ and maximum positive externality $g_{max} x_{max} $
# $$\mu_{max} = \frac{a - c}{b} + \frac{g_{max}}{b} x_{max}$$
# So the upper bound for $x_i$ is
# $$x_{max} = \frac{a - c}{b} + \frac{g_{max}}{b} x_{max} + \frac{3}{\sqrt{\beta b}} \\
# x_{max} = \frac{\frac{a - c}{b} + \frac{3}{\sqrt{\beta b}}}{1 - \frac{g_{max}}{b}}$$
#
# We then partition the interval $[0, x_{max}]$ into $K_s$ equally divided into bins and the continuous state $x_i$ is mapped to a discrete state index $S_i \in \{1, \dots, K_s\}$.
#
# ### 2. Reward Function
#
# The objective of the monopolist at time $t$ for user $i$ is the maximization of the profit. The reward $r_{i,t}$ is 
# $$r_{i,t} = (p_{i,t} - c)x_{i,t} - \kappa (\Delta p_{i,t})^2$$
# and for total reward we use an exponential discounting.
#
# ### 3. SARSA Learning Algorithm
#
# To learn the optimal strategy for the monopolist, we use *SARSA* algorithm. For each user $i$, the monopolist uses a Q-value function (discretized) $Q_i(S, A)$ representing the expected discounted future return of taking action $A$ in state $S$ and following the current policy. 
#
# Action is selected by using an $\epsilon$-greedy policy to balance exploration and exploitation
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

include("discrete_control.jl")

N_users = 5
a = 10.0
b = 2.0
c = 1.0
beta = 5.0
kappa = 0.5
s_val = fill(0.3, N_users)

G_matrix = rand(N_users, N_users) .* 0.2
G_matrix[diagind(G_matrix)] .= 0.0 

x_init = rand(N_users) .* 2.0
p_init = fill(3.0, N_users)

K_s = 5
K_a = 5

g_max = maximum(sum(G_matrix, dims=2))
x_max = (((a - c) / b) + (3.0 / sqrt(beta * b))) / (1.0 - (g_max / b))

delta_p_max = (a - c) * 0.10
actions = collect(range(-delta_p_max, delta_p_max, length=K_a))

println("Limite x_max calcolato: ", round(x_max, digits=2))
println("Azioni Δp calcolate:    ", round.(actions, digits=3))

println("\nAvvio addestramento SARSA (Epsilon fisso a 0.1)...")
Q_learned, x_history, p_history = train_sarsa!(x_init, p_init, G_matrix, a, b, c, beta, s_val, kappa, actions, x_max, K_s, epsilon=0.1)

println("Addestramento completato.")
# -

# ## <center>Reinforcement Learning: Policy Gradient</center>
#
# ### 1. One Step Actor Critic
#
# In this section we address the problem in discrete time leaving the state and action spaces continuous. With policy gradient methods one chooses a parametrized probability distribution as a policy and lets evolve its parameters to find the optimal ones, according to the Policy Gradient Theorem. Along the way a parametrized value function is learned as well. There exists a variety of policy gradient algorithms and many ways to parametrize continuous spaces. The results are particularly interesting as the monopolist doesn't have to know the topology of the network to find reach optimal prices. In this experiment the monopolist reads the consumers' usage and assigns a unique price to all of them. This initial approximation is reasoonable and justified for undirected graphs. The price will follow a Gaussian distribution in which we learn the mean. The method could be extended to learn some parameters for the variance with some refinements on the space parametrization. We follow the algorithm of One Step Actor Critic from "Reinforcement Learning: an Introduction" by R.S. Sutton and  A.G. Barto.
#
# + $\textbf{Polynomial parametrization}$: in most of the problems involving reinforcement learning on large state or action spaces (e.g. continuous spaces) one needs to implement function approximation tecniques in that, usually, only a small portion of states and/or can be actively sampled through episodes, hence the need to approximate quantities of interest - such as the value function or the action value function of the system - in such a way to generalize previous observations of the quantities themselves. In the special case of OSAC, for example, the role of the value function is essential, in that it allows the actor (i.e. the parameter of the policy) to adjust itself in a meaningful in the process of learning. The approximate value function $\hat{v}(s, \boldsymbol{w})$ associated to a given policy is, in general, expressed as a function of both the state and a vector $\boldsymbol{w}$ of parameters, which is precisely the quantity that the OSAC algorithm allows to learn; one of the most simple and immediate forms of $\hat{v}(s, \boldsymbol{w})$ is the linear one, where $\hat{v}(s, \boldsymbol{w}) = \boldsymbol{w}^{\top} \boldsymbol{x}(s)$ and $\boldsymbol{x}(s)$ represents the so-called $\textit{state feataure vector}$ associated to the state $s$, which has the same dimensions of $\boldsymbol{w}$ and has components $x_{i}(s) : \mathcal{S} \rightarrow \mathbb{R}$. The parametrization of the state feature vector can be chosen with a certain degree of freedom and multiple choices are commonly used in literature; here we provide the simplest one called "polynomial parametrization", according to which $x_{i}(s) = \prod_{j=1}^{k} s_{j}^{c_{i,j}}$ for every state $s \in \mathbb{R^{k}}$ where $c_{i,j} \in {0, 1, ..., n}$ for an arbitrary degree $n$ of the approximation. This choice brings, however, important drawbacks in the design of the RL algorithm, in that polynomial parametrization often allows a slow learning of the parameters and proves itself to be quite inefficient in the case of state spaces of large dimension. Other parametrizations can be used in order to improve the learning curve of the value function, such as the ones based on Fourier basis, tile coding or coarse coding. 
#
# + $\textbf{One Step Actor-Critic algorithm}$: the OSAC learning algorithm requires the inizialization of a random differentiable stochastic policy parametrization for $\pi(a|s, \boldsymbol{\theta})$ where $\theta$ represents a vector of learning parameters according to which mean and variance of $\pi(a|s, \boldsymbol{\theta})$ are expressed (in the following we will use the following representation of the mean alone, i.e. $\mu(\boldsymbol{\theta}) = \boldsymbol{\theta}^{\top} \boldsymbol{x}(s)$, while keeping a constant value of the variance $\sigma$ for simplicity even though, ideally, policy gadient methods aim to a progressive reduction of $\sigma$ throughout the learning process in order to obtain a fully deterministic policy in the end); furthermore an initial differentiable parametrization for the value function $\hat{v}(s, \boldsymbol{w})$ is needed. Multiple episodes are generated and, during each step of a given episode, both $\boldsymbol{\theta}$ and $\boldsymbol{w}$ are updated: in particular, after extractiong an action $A_{t}\sim\pi$ and observing a state $S'$ and a reward $R$, the parameters are updated as
# $$\delta \leftarrow R + \gamma\hat{v}(S', \boldsymbol{w})-\hat{v}(S, \boldsymbol{w})$$
# $$\boldsymbol{w} \leftarrow \boldsymbol{w} + \alpha^{\boldsymbol{w}}\delta\nabla\hat{v}(S, \boldsymbol{w})$$
# $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}}I\delta\nabla \log\pi(A|S, \boldsymbol{\theta})$$
# $$I \leftarrow \gamma I$$
# $$S \leftarrow S'$$
# where $\alpha_{\boldsymbol{\theta}}$ and $\alpha_{\boldsymbol{w}}$ are respectively the learning rates for $\boldsymbol{\theta}$ and ${\boldsymbol{w}}$. At each step, therefore, the parameters of the value function adjust themselves in the direction of the gradient of the estimated value function itself proportionally to $\delta$ (i.e. towards a target which coincides with the TD(0) method one) while the parameters of the policy adjust themselves in order to encourage the policy towards high-value states according to the estimates. At the end of such process it is expected that the parameters update lead to an optimal policy. 
#
# + $\textbf{"Noisy" Best Response}$: In order to introduce a non-linear dynamics the discrete BR relaxation behavior used in point 3 has been modified by rendering it "noisy". Namely, while every other player evolves according to a discrete BR dynamics, at each time step one player is randomly selected and a random additive component is added to its comsumption state. Despite not being a standard noisy best response dynamics (which would instead require the state to be extracted from a well specified Boltzmann distribution and would also ensure the convergence of the system towards an absolute maximum of the utility of each player) such noise still provides a fair stochastic component to the otherwise deterministic dynamics that is able to allow more exploration. 
#
# We implement the algorithms in file `one_step_actor_critic.jl`. We initialize a graph

# +
using LinearAlgebra
using Distributions
using Plots

include("myfunctions.jl")

"""INIT GRAPH"""

N = 6

G = ones(N,N)
G -= Diagonal(G)

a = 13.0
b = 12.0
c = 1.0

checkParameters(G, a, b, c)

Λ = 0.1*diagm(ones(N)+0.25*rand(N)-0.25*rand(N))


"""PRELIMINARY STUFF"""

M = influenceMatrix(G,b)
state_star = bestResponse(M, a, b, c)
println("State Star is : ", round.(state_star, digits = 4))

p_star = bestPrice(M, a, b, c)
println("Optimal Price is : ", round.(p_star, digits = 4))
# -

# We train the monopolist on this graph. We keep track of `state_star` and corresponding optimal price hoping that the agent will learn to lead the discrete time dynamic towards that situation. The parameters choice is essential to converge to a stable result.

# +
include("One_step_Actor_Critic.jl")

"""EPISODE PARAMETER"""
γ = 0.90
T = 100

"""LEARNING PARAMETERS"""
σ = 0.25
α_θ = 5e-7
α_w = 1e-5
num_training = 10_000

my_θ = one_step_actor_critic(α_θ, α_w, γ, σ, T, num_training);
# -

# We can now generate an episode with the learned parameters $ \theta $. The monopolist seems to have understood where to drive the consumers regardless of the initial conditions imposed.

# +
include("plot_functions.jl")

"""GENERATE EPISODE"""

state_0 = rand(Normal(0.9, 0.1), N)

my_states, my_prices, my_rewards = generate_episode(my_θ, state_0, T);

plot_discrete_episode(my_states, my_prices, 1, T)
# -

# The results seem rather good. We compare them with episode generated starting from the same initial conditions but according to a relaxation dynamics in which the monopolist controls the prices with an artificial drive towards the optimal price at the Nash equilibrium (p_star).

eu_states, eu_prices, eu_rewards = generate_episode(my_θ, state_0, T; euristic=true);
plot_discrete_episode(eu_states, eu_prices, 1, T)

# Cumulative rewards with the two methods are on average very similar, the monopolist learned well with the OSAC algorithm. Whereas the euristic exponential relies on the knowledge of the network to characterize Nash Equilibria, the policy learned thanks to policy gradient does not and, even in such simple approximation where the monopolist has been trained to assign the same price to every player, the algorithm has proven itself able to recognize the N.E on an undirected graph (a situation in which, by construction, the topolocy of the graph brings to the well known result of common prices for every user) as the best possible stratedy to use on the long run.

# ## <center>Conclusions</center>
#
# Congrats to the team.
