"""CONTINUOUS TIME CONTROL"""

function calculate_reward(times::Vector{}, X::Matrix{}, P::Matrix{}, U::Matrix{}, c, 𝛿, k)

    Δt = times[2]-times[1]
    steps = length(times)

    cumulative_reward = 0.0
    rewards = zeros(steps)

    for s in 1:steps
        cumulative_reward += exp(-𝛿*times[s])*(transpose(X[:,s])*P[:,s] - 0*c*sum(X[:,s]) - k*0.5*sum(U[:,s].^2))*Δt
        rewards[s] = cumulative_reward
    end

    println("Cumulative reward: $(round(cumulative_reward, digits=2))")
    rewards
end

function integrate_backward(A::Matrix{}, B::Matrix{}, D::Vector{}, K::Matrix{}, N, 𝛿, Δt, steps)

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

    println("BACKWARD INTEGRATION TERMINATED SUCCESSFULLY")
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
    
    end

    println("FORWARD INTEGRATION TERMINATED SUCCESSFULLY")

    return X,P,U
end