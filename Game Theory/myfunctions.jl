using SparseArrays, LinearAlgebra
using Graphs

"""GRAPH CREATORS"""

lattice(n) = diagm(1=>trues(n-1), -1=>trues(n-1)) 
clattice(n) = n == 2 ? lattice(n) : diagm(1=>trues(n-1), -1=>trues(n-1), (n-1)=>trues(1), (-n+1)=>trues(1))

function createGraphLattice(N1,N2; periodic = false)
    kron(I(N2),lattice(N1)) .| kron(lattice(N2),I(N1))
end

function createUndirectedGraphStar(N)
    G = zeros(N,N)
    G[1,2:N] = 1.0
    G[2:N,1] = 1.0
    G 
end

function createDirectedGraphStar(N)
    G = zeros(N,N)
    G[2:N,1] = 1.0
    G 
end

function createDirectedGraphRing(N)
        G = zeros(N, N)
        for i in 1:N-1
            G[i, i+1] = 1.0
        end
    G[N, 1] = 1.0   
    G 
end

function createGraphErdos(N; p=0.1,sparse=true)

    G = adjacency_matrix(erdos_renyi(N, N))      #creates Erdos Renyi graph with random weights
    #G = (UpperTriangular(adjacency_matrix(erdos_renyi(N, 0.5*p))) + LowerTriangular(adjacency_matrix(erdos_renyi(N, 0.5*p))))
    G -= Diagonal(G)
    #G = G .* rand(N,N)
    sparse ? G : Matrix(G)                                  #Maybe we want sparse for later optimization

end

function createGraphFullyConnected(N)
    G = ones(N,N)
    G -= Diagonal(G)
    G
end

function createGraphInfluencersCommunities(N_infl, size_com, n_com)
    N = N_infl + n_com*size_com
    G = zeros(N, N)
    for i in 0:n_com-1
        G[N_infl+1+size_com*i:N_infl+size_com+i*size_com, N_infl+1+size_com*i:N_infl+size_com+i*size_com] = createGraphFullyConnected(size_com)
    end
    G[N_infl+1:N, 1:N_infl] .= 1.0
    G
end

function checkParameters(G, a, b, c)
    if a < 0 || b < 0 || c < 0
        println("Please provide non negative parameters")
    elseif c ≥ a 
        println("Please check a > c")
    elseif b < maximum(sum(G;dims=2))
        println("Parameter b is low. Multiple equilibria may occur.")
    else
        println("Correctly Initialized Network")
    end
end

function utility(x, p, G, a, b)
    u = a*x - 0.5*b*(x.^2) + x.*(G*x) - p.*x 
end

function reward(x, p, c)
    transpose(p.-c)*x
end

function influenceMatrix(G, b)
    M = inv(I - G/b)
end

function bestResponse(M, a, b, c)
    return (I - M * inv(M + transpose(M)))*M*((a-c)/b)*ones(size(M)[1])
end

function bestPrice(M, a, b, c)
    return inv(M + transpose(M))*(a*M+c*transpose(M))*ones(size(M,2))
end

"""CONTINUOUS TIME CONTROL"""

function calculate_reward(times::Vector{}, X::Matrix{}, P::Matrix{}, U::Matrix{}, c, 𝛿, k)

    Δt = times[2]-times[1]
    steps = length(times)

    cumulative_reward = 0.0
    reward = zeros(steps)

    for s in 1:steps
        cumulative_reward += exp(-𝛿*times[s])*(transpose(X[:,s])*P[:,s] - 0*c*sum(X[:,s]) - k*0.5*sum(U[:,s].^2))*Δt
        reward[s] = cumulative_reward
    end

    reward
end

function integrate_backward(A::Matrix{}, B::Matrix{}, D::Vector{}, g::Vector{}, K::Matrix{}, N, 𝛿, Δt, steps)

    Q = [0I(N) I(N);
         I(N) 0I(N)]

    S = fill(zeros(2*N,2*N),steps)               #Vector of matrices!
    v = zeros(2*N,steps)

    #Final conditions already fulfilled

    for s in (steps-1):-1:1

        S_t = S[s+1]

        dS = (-transpose(A)*S_t - S_t*A + 𝛿*S_t + S_t*B*inv(K)*transpose(B)*S_t + Q)*Δt    
        dv = ((S_t*B*inv(K)*transpose(B) - transpose(A) + 𝛿*I(2*N))*v[:,s+1] - S_t*D)*Δt

        S[s] = S[s+1] - dS
        v[:,s] = v[:,s+1] - dv

    end

    return S,v
end

function integrate_forward(x0::Vector{}, p0::Vector{}, S::Vector{}, v::Matrix{}, G::Matrix{}, K::Matrix{}, Λ::Matrix{}, B::Matrix{}, N, a, b, c, Δt, steps)
    
    aI = a*ones(N)

    X = zeros(N,steps)
    P = zeros(N,steps)

    U = zeros(N,steps)

    X[:,1] = x0
    P[:,1] = p0

    #println("@ s = 1 X: " , X[:,1])
    #println("@ s = 1 P: " , P[:,1])
    #U = zeros(N,steps);

    for s in 2:steps

        x_t = X[:,s-1]
        p_t = P[:,s-1]

        S_t = S[s-1]
        v_t = v[:,s-1]

        X_dot = -Λ*x_t + Λ*(aI - p_t + G*x_t)/b
        #X_dot = -Λ*x_t + Λ*(aI - p_star + G*x_t)/b
        P_dot = -inv(K)*transpose(B)*(S_t*vcat(x_t, p_t) + v_t)

        X[:,s] = X[:,s-1] + X_dot*Δt
        P[:,s] = P[:,s-1] + P_dot*Δt

        U[:,s] = P_dot

        """
        if s == 2
            println("@ s = 2 X: " , X[:,2])
            println("@ s = 2 P: " , P[:,2])
        elseif mod(s,10)==0
            println("@ s = $s X: " , X[:,s])
            println("@ s = $s P: " , P[:,s])
        end
        """
    
    end

    return X,P,U
end


