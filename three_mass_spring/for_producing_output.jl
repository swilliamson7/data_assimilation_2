using Enzyme, LinearAlgebra
using Parameters, UnPack

@with_kw mutable struct mso_params_ops{F<:Function}

    T::Int               # Total steps to integrate 
    t::Int = 0           # placeholder for current step
    dt::Float64 = 0.001  # timestep

    x::Vector{Float64} = zeros(6)      # placeholder for state vector 
    u::Matrix{Float64} = zeros(6, T+1)   # random forcing (if any), otherwise just leave zero 
    n::Matrix{Float64} = zeros(6, T+1)   # placeholder for noise to add to the data 

    k::Int = 30          # spring constant
    r::Float64 = 0.5     # Rayleigh friction coefficient

    q::F                 # forcing function 

    J::Float64 = 0.0     # cost function evaluation 

    data_steps::Vector{Int64}       # the timesteps where data points exist 
    data::Matrix{Float64}

    states::Matrix{Float64}    # placeholder for computed states 
    energy::Matrix{Float64}    # placeholder for computed energy

    A::Matrix{Float64} = zeros(6,6)                            # Time-step (x(t+1) = A x(t))
    B::Matrix{Float64} = diagm([1., 0., 0., 0., 0., 0.])       # Distributes known forcing  
    Gamma::Matrix{Float64} = zeros(6,6)                        # Distrubutes unknown (random) forcing

    E::Matrix{Float64} = zeros(6,6)           # Acts on data vector, generally the identity (e.g. full info on all positions/velocities)

    Q::Matrix{Float64} = zeros(6,6)           # Covariance matrix for unknown (random) forcing 
    R::Matrix{Float64} = zeros(6,6)           # Covariance matrix for noise in data 

    K::Matrix{Float64} = zeros(6,6)           # Placeholder for Kalman gain matrix 
    Kc::Matrix{Float64} = zeros(6,6)     

end

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

function integrate(mso_struct)

    @unpack A, B, Gamma, E, R, Kc, Q = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct 


    Q_inv = 1/Q[1,1]
    temp = 1 ./ diag(R) 

    R_inv = [temp[1] 0. 0. 0. 0. 0.; 
        0. temp[2] 0. 0. 0. 0.; 
        0. 0. temp[3] 0. 0. 0.; 
        0. 0. 0. temp[4] 0. 0.; 
        0. 0. 0. 0. temp[5] 0.; 
        0. 0. 0. 0. 0. temp[6]
    ]

    # run the model forward to get estimates for the states 
    temp = 0.0 
    for t = 2:T+1 

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t]
        states[:, t] .= copy(x) 

        temp += dt

        if t in data_steps 

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R_inv * (E * x - E * data[:, t]) + u[:, t]' * Q_inv * u[:, t]

        end


    end 

    return nothing 

end

# Parameter choices 
T = 10000          # Total number of steps to integrate
r = 0.5
k = 30
dt = .001
q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
q_kf(t) = 0.5 * q_true(t)                                           # forcing seen by KF (and adjoint)
data_steps = [0]                         # steps where data will be assimilated

rand_forcing = 0.1 .* ones(T+1)
u = zeros(6, T+1)
u[1, :] .= rand_forcing

Rc = -r .* diagm(ones(3))
# Kc = [-2*k k 0; k -3*k k; 0 k -2*k]
Kc = zeros(3,3)
# Ac = [zeros(3,3) diagm(ones(3))
#      Kc Rc
# ]
Ac = zeros(6,6)

A = diagm(ones(6)) + dt .* Ac

ops = mso_operators(A=A, 
Kc=Kc
)

# assuming data of all positions and velocities -> E is the identity operator 
ops.E .= Diagonal(ones(6))

ops.Q[1,1] = 1.0 #cov(u[:], corrected=false)
ops.R .=  Diagonal(ones(6))

# assuming random forcing on position of mass one 
ops.Gamma[1, 1] = 1.0 

# with Enzyme

params_adjoint = mso_params_ops(T=T, 
    t = 0,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = 0.0 .* u,
    n = 0.001 .* ones(6,T+1),
    q = q_kf,
    data_steps = data_steps,
    data = ones(6,2),
    states = zeros(6, T+1),
    energy = zeros(3, T+1), 
    A = ops.A,
    B = ops.B, 
    Gamma = ops.Gamma, 
    E = ops.E, 
    Q = ops.Q, 
    R = ops.R,
    K = ops.K,
    Kc = ops.Kc
)

dparams = deepcopy(params_adjoint)

autodiff(Reverse, integrate, Duplicated(params_adjoint, dparams))

# integrate(params_adjoint)