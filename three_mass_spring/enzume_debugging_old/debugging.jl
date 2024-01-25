using Enzyme#main
using LinearAlgebra, Statistics, Random
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

@with_kw mutable struct mso_params{F<:Function}

    T::Int               # Total steps to integrate 
    dt::Float64 = 0.001  # timestep

    x::Vector{Float64} = zeros(6)      # placeholder for state vector 
    u::Matrix{Float64} = zeros(6, T+1)   # random forcing (if any), otherwise just leave zero 
    n::Matrix{Float64} = zeros(6, T+1)   # placeholder for noise to add to the data 

    k::Int = 30          # spring constant
    r::Float64 = 0.5     # Rayleigh friction coefficient

    q::F                 # forcing function 

    J::Float64 = 0.0     # cost function storage

    data_steps::Vector{Int64}       # the timesteps where data points exist 
    data::Matrix{Float64}

    states::Matrix{Float64}    # placeholder for computed states 
    energy::Matrix{Float64}    # placeholder for computed energy

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

# computes the kinetic energy given a state vector x 
function compute_energy(x, Kc)

    kin = 0.5 * (x[4:6]' * x[4:6])
    ptl = 0.5 * (-x[1:3]' * Kc * x[1:3])

    return kin, ptl, kin + ptl

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

        x[:] = A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, j]

        states[:, j] .= copy(x)

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,j] = [kin;ptl;total]

        states_noisy[:, j] = copy(x) + n[:, j]
        temp += dt

    end 

    return states_noisy

end

function integrate(mso_struct, q::Function)

    @unpack A, B, Gamma, E, R, Kc, Q = mso_struct
    @unpack T, x, u, dt, states, data, energy =  mso_struct
    @unpack data_steps, J = mso_struct 

    # kin, ptl, total = compute_energy(states[:,1], Kc)
    # energy[:,1] = [kin;ptl;total]

    Q_inv = 1/Q[1,1]

    # run the model forward to get estimates for the states 
    temp = 0.0 
    for t = 2:T+1 

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t]
        states[:, t] .= copy(x) 

        temp += dt

        if t in data_steps 

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * (E * x - E * data[:, t]) 

            if t != T+1 && t in data_steps 
                mso_struct.J = mso_struct.J + u[:, t]' * Q_inv * u[:, t]
            end

        end
    end 

    return nothing 

end

function integrate(mso_struct)

    @unpack A, B, Gamma, E, R, Kc, Q = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct 

    # kin, ptl, total = compute_energy(states[:,1], Kc)
    # energy[:,1] = [kin;ptl;total]

    Q_inv = 1/Q[1,1]

    # run the model forward to get estimates for the states 
    temp = 0.0 
    for t = 2:T+1 

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t]
        states[:, t] .= copy(x) 

        temp += dt

        if t in data_steps 

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * (E * x - E * data[:, t]) 

            if t != T+1 && t in data_steps 
                mso_struct.J = mso_struct.J + u[:, t]' * Q_inv * u[:, t]
            end

        end
    end 

    return nothing 

end

 # Parameter choices 
 T = 10000          # Total number of steps to integrate
 r = 0.5
 q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
 q_kf(t) = 0.5 * q_true(t)                                           # forcing seen by KF (and adjoint)
 data_steps = [1500 + k*1000 for k in 1:5]      # steps where data will be assimilated

 rand_forcing = 0.1 .* randn(T+1)
 u = zeros(6, T+1)
 u[1, :] .= rand_forcing

 params_true = mso_params(T = T,
 x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 u = u,
 n = 0.05 .* randn(6, T+1), 
 q = q_true,
 data_steps = data_steps,
 data = zeros(1,1),
 states = zeros(6, T+1),
 energy = zeros(3, T+1)
 )

 params_pred = mso_params(T = T,
 x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
 q = q_kf, 
 data_steps = data_steps,
 data = zeros(1,1),
 states = zeros(6, T+1), 
 energy = zeros(3, T+1)
 )

 ops = build_ops(params_true)

 # assuming data of all positions and velocities -> E is the identity operator 
 ops.E .= Diagonal(ones(6))

 ops.Q[1,1] = cov(params_true.u[:], corrected=false)
 ops.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

 # assuming random forcing on position of mass one 
 ops.Gamma[1, 1] = 1.0 

 # create data from the true setup
 states_noisy = create_data(params_true, ops)

 params_adjoint = mso_params_ops(T=T, 
     t = 0,
     x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     u = 0.0 .* u,
     n = 0.001 .* randn(6, T+1), 
     q = q_kf,
     data_steps = data_steps,
     data = states_noisy,
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
# integrate(params_adjoint, q_kf)

autodiff(Reverse, integrate, Duplicated(params_adjoint, dparams))

 # snaps = 1 
 # verbose = 1 
 # revolve = Revolve{mso_params_ops}(params_adjoint.T, snaps; verbose=verbose)

 # dparams = Zygote.gradient(
 #     cp_adjoint, 
 #     params_adjoint,
 #     revolve,
 #     q_kf
 # )