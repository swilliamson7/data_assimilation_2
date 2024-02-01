using Plots, Enzyme, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings
using Checkpointing

Enzyme.API.runtimeActivity!(true)

# include("misc_functions.jl")
# include("kalman_filter.jl")
# include("adjoint.jl")

function build_ops(params; E = zeros(6,6), R = zeros(6,6), K = zeros(6,6))
    
    dt = params.dt
    k = params.k
    r = params.r

    Rc = -r .* diagm(ones(3))
    Kc = [-2*k k 0; k -3*k k; 0 k -2*k]
    Ac = [zeros(3,3) diagm(ones(3))
         Kc Rc
    ]
    
    A = diagm(ones(6)) + dt .* Ac
    
    ms_ops = mso_operators(A=A,
    E=E,
    R=R,
    K=K,
    Kc=Kc
    )

    return ms_ops
end

function create_data(params, ops)

    @unpack T, x, u, n, q, dt, states, energy = params
    @unpack A, B, Gamma, Kc = ops

    states[:,1] .= x

    states_noisy = zeros(6,T+1)
    states_noisy[:,1] .= x + n[:,1]
    kin, ptl, total = compute_energy(states[:,1], Kc)
    energy[:, 1] = [kin;ptl;total]

    temp = 0.0
    for j = 2:T+1

        x[:] = A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, j-1]
        states[:, j] .= copy(x)

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,j] = [kin;ptl;total]

        states_noisy[:, j] .= copy(x) + n[:, j]
        temp += dt

    end 

    return states_noisy

end

function compute_energy(x, Kc)

    kin = 0.5 * (x[4:6]' * x[4:6])
    ptl = 0.5 * (-x[1:3]' * Kc * x[1:3])

    return kin, ptl, kin + ptl

end

@with_kw mutable struct mso_params{F<:Function}

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

@with_kw mutable struct mso_params_ops{F<:Function}

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

# structure containing both time-stepping operators as well as operators
# related to the Kalman filter operators (e.g. the Kalman gain matrix K)
@with_kw struct mso_operators

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

function integrate2(mso_struct::mso_params_ops, scheme)

    @unpack B, Gamma, E, R, R_inv, Kc, Q, Q_inv = mso_struct
    @unpack r = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct

    t = mso_struct.t

    k = mso_struct.k

    states[:,1] .= x

    kin, ptl, total = compute_energy(states[:,1], Kc)
    energy[:,1] = [kin;ptl;total]

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
    @checkpoint_struct scheme mso_struct for mso_struct.t = 2:T+1

        x[:] = A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:,mso_struct.t-1]

        temp += dt

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

        if t in data_steps

            mso_struct.J = mso_struct.J + sum((energy[:,t]).^2)

        end

    end

    return states[1, T]

end

function run()

    # Parameter choices
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient
    k_guess=25.

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    ###########################################

    data_steps1 = [k for k in 3000:200:7000]         # steps where data will be assimilated
    data_steps = data_steps1

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = mso_params(T = T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        k = 30,
        n = 0.01 .* randn(6, T+1),
        q = q_true,
        data_steps = data_steps,
        data = zeros(1,1),
        states = zeros(6, T+1),
        energy = zeros(3, T+1)
    )

    params_pred = mso_params(T = T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        q = q_kf,
        k = k_guess,
        data_steps = data_steps,
        data = zeros(1,1),
        states = zeros(6, T+1),
        energy = zeros(3, T+1)
    )

    ops_true = build_ops(params_true)
    ops_pred = build_ops(params_pred)

    # assuming data of all positions and velocities -> E is the identity operator
    ops_true.E .= Diagonal(ones(6))
    ops_pred.E .= Diagonal(ones(6))

    ops_true.Q[1,1] = cov(params_true.u[:], corrected=false)
    ops_pred.Q[1,1] = cov(params_true.u[:], corrected=false)

    ops_true.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))
    ops_pred.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops_true.Gamma[1,1] = 1.0
    ops_pred.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = create_data(params_pred, ops_pred)

    states_noisy = create_data(params_true, ops_true)

    diag = 1 / ops_pred.Q[1,1]
    Q_inv = diag

    ###############################
    R_inv = ops_pred.R^(-1)
    # R_inv = ops_pred.E
    ###############################

    params_adjoint = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = 0.0 .* u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k_guess,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops_pred.A,
        B = ops_pred.B,
        Gamma = ops_pred.Gamma,
        E = ops_pred.E,
        Q = 0.0 .* ops_pred.Q,
        Q_inv = Q_inv,
        R = ops_pred.R,
        R_inv = R_inv,
        K = ops_pred.K,
        Kc = ops_pred.Kc
    )

    dparams_adjoint = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
    dparams_adjoint.J = 1.0
    dparams_adjoint.k = 0.
    dparams_adjoint.r = 0.
    dparams_adjoint.dt = 0.
    dparams_adjoint.Q_inv = 0.

    snaps = Int(floor(sqrt(T)))
    revolve = Revolve{mso_params_ops}(T,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false
    )

    autodiff(Enzyme.ReverseWithPrimal,
        integrate2,
        Duplicated(params_adjoint, dparams_adjoint),
        revolve
    )

end

run()