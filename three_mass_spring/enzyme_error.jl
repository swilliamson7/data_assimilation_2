using Enzyme
using Random
using Parameters
using LinearAlgebra

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

    states::Matrix{Float64}    # placeholder for computed states

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

function integrate(mso_struct::mso_params_ops)

    @unpack A, B, Gamma, E, R, R_inv, Kc, Q, Q_inv = mso_struct
    @unpack T, x, u, dt, states,q =  mso_struct
    @unpack J = mso_struct

    states[:,1] .= x

    # run the model forward to get estimates for the states 
    temp = 0.0
    for t = 2:T+1

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
        states[:, t] .= copy(x)

        temp += dt

    end

    return nothing 

end

function error1()

    # Parameter choices 
    T = 10000          # Total number of steps to integrate
    r = 0.5
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
    q_kf(t) = 0.5 * q_true(t)                                           # forcing seen by KF (and adjoint)
    data_steps = [1500 + k*1000 for k in 1:5]      # steps where data will be assimilated

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    # diag = 1 / ops.Q[1,1]
    Q_inv = 1
    # R_inv = ops.R^(-1)
    params_adjoint = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = 0.0 .* u,
        n = 0.001 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        states = zeros(6, T+1),
        Q_inv = Q_inv,
        R_inv = Diagonal(ones(6))
    )

    dparams = Enzyme.make_zero(params_adjoint)
    dparams.J = 1.0

    autodiff(Reverse, integrate, Duplicated(params_adjoint, dparams))

end

error1()