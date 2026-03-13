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
    G[1,2:N] .= 1.0
    G[2:N,1] .= 1.0
    G 
end

function createDirectedGraphStar(N)
    G = zeros(N,N)
    G[2:N,1] .= 1.0
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


