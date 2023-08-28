using Enzyme, LinearAlgebra, Statistics, Random
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

function integrate(mso_struct)

    @unpack A, B, Gamma, E, R, Kc, Q = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct 

    Q_inv = 1/Q[1,1]

    # run the model forward to get estimates for the states 
    temp = 0.0 
    for t = 2:T+1 

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t]
        states[:, t] .= copy(x) 

        temp += dt

        if t in data_steps 

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R^(-1) * (E * x - E * data[:, t]) 

            if t != T && t in data_steps 
                mso_struct.J = mso_struct.J + u[:, t]' * Q_inv * u[:, t]
            end

        end

    end 

    return nothing 

end

# Parameter choices 
T = 10000          # Total number of steps to integrate
r = 0.5
q(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
data_steps = [1500 + k*1000 for k in 1:5]      # steps where data will be assimilated
dt = 0.001
k = 30

rand_forcing = 0.1 .* randn(T+1)
u = zeros(6, T+1)
u[1, :] .= rand_forcing
n = 0.05 .* randn(6, T+1)

E = zeros(6,6)
R = zeros(6,6)
Q = zeros(6,6)
K = zeros(6,6)
Gamma = zeros(6,6)
Rc = -r .* diagm(ones(3))
Kc = [-2*k k 0; k -3*k k; 0 k -2*k]
Ac = [zeros(3,3) diagm(ones(3))
     Kc Rc
]

A = diagm(ones(6)) + dt .* Ac

Q[1,1] = cov(u[:], corrected=false)
R .= cov(n[:], corrected=false) .* Diagonal(ones(6))

# assuming random forcing on position of mass one 
Gamma[1, 1] = 1.0 

# create data from the true setup
states_noisy = zeros(6, T+1)

params_adjoint = mso_params_ops(T=T, 
    t = 0,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = 0.0 .* u,
    n = 0.001 .* randn(6, T+1), 
    q = q,
    data_steps = data_steps,
    data = states_noisy,
    states = zeros(6, T+1),
    energy = zeros(3, T+1), 
    A = A,
    B = diagm([1., 0., 0., 0., 0., 0.]), 
    Gamma = diagm([1., 0., 0., 0., 0., 0.]), 
    E = Diagonal(ones(6)), 
    Q = Q, 
    R = R,
    K = K,
    Kc = Kc
)

integrate(params_adjoint)

dparams = deepcopy(params_adjoint)
# autodiff(Reverse, integrate, Duplicated(params_adjoint, dparams))