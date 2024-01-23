using Optimization, OptimizationOptimJL
using Enzyme, LinearAlgebra, Statistics, Random
using Parameters, UnPack

@with_kw mutable struct mso_params_ops1{F<:Function}

    T::Int               # Total steps to integrate 
    t::Int = 0           # placeholder for current step
    dt::Float64 = 0.001  # timestep

    x::Vector{Float64} = zeros(6)        # placeholder for state vector
    u::Matrix{Float64} = zeros(6, T+1)   # random forcing (if any), otherwise just leave zero
    n::Matrix{Float64} = zeros(6, T+1)   # placeholder for noise to add to the data

    k::Float64 = 30          # spring constant
    r::Float64 = 0.5         # Rayleigh friction coefficient

    q::F                     # forcing function

    J::Float64 = 0.0         # cost function evaluation 

    data_steps::Vector{Int64}  # the timesteps where data points exist 
    data::Matrix{Float64}

    states::Matrix{Float64}    # placeholder for computed states
    energy::Matrix{Float64}    # placeholder for computed energy

    A::Matrix{Float64} = zeros(6,6)                            # Time-step (x(t+1) = A x(t))
    B::Matrix{Float64} = diagm([1., 0., 0., 0., 0., 0.])       # Distributes known forcing
    Gamma::Matrix{Float64} = zeros(6,6)                        # Distrubutes unknown (random) forcing

    E::Matrix{Float64} = zeros(6,6)           # Acts on data vector, generally the identity (e.g. full info on all positions/velocities)

    Q::Matrix{Float64} = zeros(6,6)           # Covariance matrix for unknown (random) forcing
    Q_inv::Float64
    R::Matrix{Float64} = zeros(6,6)           # Covariance matrix for noise in data
    R_inv::Matrix{Float64}                    # Inverse of operator R

    K::Matrix{Float64} = zeros(6,6)           # Placeholder for Kalman gain matrix 
    Kc::Matrix{Float64} = zeros(6,6)     

end

@with_kw mutable struct mso_params1{F<:Function}

    T::Int               # Total steps to integrate 
    dt::Float64 = 0.001  # timestep

    x::Vector{Float64} = zeros(6)      # placeholder for state vector 
    u::Matrix{Float64} = zeros(6, T+1)   # random forcing (if any), otherwise just leave zero 
    n::Matrix{Float64} = zeros(6, T+1)   # placeholder for noise to add to the data 

    k::Float64 = 30          # spring constant
    r::Float64 = 0.5     # Rayleigh friction coefficient

    q::F                 # forcing function 

    J::Float64 = 0.0     # cost function storage

    data_steps::Vector{Int64}       # the timesteps where data points exist
    data::Matrix{Float64}

    states::Matrix{Float64}    # placeholder for computed states
    energy::Matrix{Float64}    # placeholder for computed energy

end

@with_kw struct mso_operators1

    A::Matrix{Float64} = zeros(6,6)                            # Time-step (x(t+1) = A x(t))
    B::Matrix{Float64} = diagm([1., 0., 0., 0., 0., 0.])       # Distributes known forcing  
    Gamma::Matrix{Float64} = zeros(6,6)                        # Distrubutes unknown (random) forcing

    P0::Matrix{Float64} = zeros(6,6)          # Init. uncertainty operator (generally only non-zero when x0 not fully known)
    P::Matrix{Float64} = zeros(6,6)           # Placeholder for future uncertainty operators  

    E::Matrix{Float64} = zeros(6,6)           # Acts on data vector, generally the identity (e.g. full info on all positions/velocities)

    Q::Matrix{Float64} = zeros(6,6)           # Covariance matrix for unknown (random) forcing
    R::Matrix{Float64} = zeros(6,6)           # Covariance matrix for noise in data 

    K::Matrix{Float64} = zeros(6,6)           # Placeholder for Kalman gain matrix 
    Kc::Matrix{Float64} = zeros(6,6)

end

function integrate1(mso_struct::mso_params_ops1)

    @unpack B, Gamma, E, R, R_inv, Kc, Q, Q_inv = mso_struct
    @unpack r = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct

    k = mso_struct.k

    states[:,1] .= x

    Ac = zeros(6,6)
    Ac[1,4] = 1
    Ac[2,5] = 1
    Ac[3,6] = 1
    Ac[4,1] = -2*k
    Ac[4,2] = k
    Ac[4,4] = -r
    Ac[5,1] = k
    Ac[5,2] = -3*k
    Ac[5,3] = k
    Ac[5,5] = -r
    Ac[6,2] = k
    Ac[6,3] = -2*k
    Ac[6,6] = -r
    A = diagm(ones(6)) + dt .* Ac

    # run the model forward to get estimates for the states
    temp = 0.0
    for t = 2:T+1

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
        states[:, t] .= copy(x)

        temp += dt

        if t in data_steps

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * (E * x - E * data[:, t])

        end

    end

    return nothing

end

function cost_function_eval(k_guess, p)

    # Parameter choices 
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))
    q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [j for j in 3000:200:7000]
    data_steps = data_steps1

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = mso_params1(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
    k = 30.,
    n = 0.0 .* randn(6, T+1), ###################################
    q = q_true,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    Rc = -r .* diagm(ones(3))
    Kc = [-2*params_true.k params_true.k 0; params_true.k -3*params_true.k params_true.k; 0 params_true.k -2*params_true.k]
    Ac = [zeros(3,3) diagm(ones(3))
         Kc Rc
    ]

    A = diagm(ones(6)) + params_true.dt .* Ac

    # assuming data of all positions and velocities -> E is the identity operator
    E = Diagonal(ones(6))

    Q = zeros(6,6)
    Q[1,1] = cov(params_true.u[:], corrected=false)
    R = cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

    # assuming random forcing on position of mass one
    Gamma = zeros(6,6)
    Gamma[1, 1] = 1.0
    
    ops = mso_operators1(A=A,
    E=E,
    R=R,
    Kc=Kc,
    Q=Q,
    Gamma=Gamma
    )

    params_true.states[:,1] .= params_true.x

    states_noisy = zeros(6,T+1)
    states_noisy[:,1] .= params_true.x + params_true.n[:,1]

    temp = 0.0
    for j = 2:T+1

        params_true.x[:] = ops.A * params_true.x + ops.B * [params_true.q(temp); 0.; 0.; 0.; 0.; 0.] + ops.Gamma * params_true.u[:, j-1]
        params_true.states[:, j] .= params_true.x

        states_noisy[:, j] .= params_true.x + params_true.n[:, j]
        temp += params_true.dt

    end

    Q_inv = 0.0

    ######################
    R_inv = ops.E
    ######################

    params_adjoint = mso_params_ops1(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        q = q_kf,
        J = 0.0,
        k = k_guess[1],
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops.A,
        B = ops.B,
        Gamma = ops.Gamma,
        E = ops.E,
        Q = 0.0 .* ops.Q,
        Q_inv = Q_inv,
        R = ops.R,
        R_inv = R_inv,
        K = ops.K,
        Kc = ops.Kc
    )

    integrate1(params_adjoint)

    return params_adjoint.J

end

k0 = [31.]
p = 0.0
optf = OptimizationFunction(cost_function_eval, Optimization.AutoEnzyme())
prob = OptimizationProblem(optf, k0, p)
sol = solve(prob, BFGS())

# dparams = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
# dparams.J = 1.0
# dparams.k = 0.
# dparams.r = 0.
# dparams.dt = 0.
# dparams.Q_inv = 0.

# Enzyme.autodiff(Reverse, integrate1, Duplicated(params_adjoint, dparams))
