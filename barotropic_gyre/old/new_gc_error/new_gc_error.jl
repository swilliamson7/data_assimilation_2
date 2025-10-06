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

    states[:,1] .= x

    Q_inv = 1/Q[1,1]
    
    temp = 1 ./ diag(R) 
    R_inv = zeros(6,6)
    R_inv[1,1] = temp[1] 
    R_inv[2,2] = temp[2] 
    R_inv[3,3] = temp[3]
    R_inv[4,4] = temp[4]
    R_inv[5,5] = temp[5]
    R_inv[6,6] = temp[6]

    # run the model forward to get estimates for the states 
    temp = 0.0 
    for t = 2:T+1 

        x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
        states[:, t] .= copy(x) 

        temp += dt

        if t in data_steps 

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R_inv * (E * x - E * data[:, t]) + u[:, t]' * Q_inv * u[:, t]

        end

    end 

    return nothing 

end

function grad_descent(M, params)

    @unpack T = params

    u_new = zeros(6, T+1)

    params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params.states .= zeros(6, T+1)
    params.energy .= zeros(3, T+1)

    dparams = mso_params_ops(T=0.0, 
        t = 0,
        dt = 0.0,
        x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = 0.0 .* u,
        n = zeros(6, T+1), 
        q = q_kf,
        k = 0.0,
        r = 0.0,
        J = 1.0,
        data_steps = [0.0],
        data = zeros(6,T+1),
        states = zeros(6, T+1),
        energy = zeros(3, T+1), 
        A = zeros(6,6),
        B = zeros(6,6), 
        Gamma = zeros(6,6), 
        E = zeros(6,6), 
        Q = zeros(6,6), 
        R = zeros(6,6),
        K = zeros(6,6),
        Kc = zeros(6,6)
    )

    autodiff(Reverse, integrate, Duplicated(params, dparams))

    for t = 1:T 
            
        u_new[:, t] = params.u[:, t] - 1 / (norm(dparams.u[1,:])) * dparams.u[:, t]

    end

    k = 1
    u_old = copy(params.u)
    u_grad_old = copy(dparams.u)
    params.u .= u_new 

    while norm(u_grad_old[1,:]) > 500

        params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        params.states .= zeros(6, T+1)

        dparams = mso_params_ops(T=0.0, 
            t = 0,
            dt = 0.0,
            x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            u = 0.0 .* u,
            n = zeros(6, T+1), 
            q = q_kf,
            k = 0.0,
            r = 0.0,
            J = 1.0,
            data_steps = [0.0],
            data = zeros(6,T+1),
            states = zeros(6, T+1),
            energy = zeros(3, T+1), 
            A = zeros(6,6),
            B = zeros(6,6), 
            Gamma = zeros(6,6), 
            E = zeros(6,6), 
            Q = zeros(6,6), 
            R = zeros(6,6),
            K = zeros(6,6),
            Kc = zeros(6,6)
        )

        autodiff(Reverse, integrate, Duplicated(params, dparams))

        @show norm(dparams.u[1,:])

        gamma = 0.0
        num = 0.0
        den = 0.0

        for j = 1:T 

            num += sum(dot(params.u[:,j] - u_old[:,j], dparams.u[:,j] - u_grad_old[:,j]))
            den += norm(dparams.u[:,j] - u_grad_old[:,j])^2

        end

        gamma = (abs(num) / den)

        for t = 1:T 

            u_new[:, t] = params.u[:, t] - gamma * dparams.u[:, t]

        end

        u_old = copy(params.u)
        u_grad_old = copy(dparams.u)
        params.u .= u_new
        dparams.u = zeros(6, T+1)

        k += 1

        if k > M
            break
        end

    end

end

T = 10000  
const r = 0.5
k = 30
dt = .001
q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                    
q_kf(t) = 0.5 * q_true(t)                                          
data_steps = [0]                                                     

u = zeros(6, T+1)


Rc = -r .* diagm(ones(3))
Kc = zeros(3,3)
Ac = zeros(6,6)

A = diagm(ones(6)) + dt .* Ac

ops = mso_operators(A=A, 
Kc=Kc
)

ops.E .= Diagonal(ones(6))
ops.Q[1,1] = 1.0 
ops.R .=  Diagonal(ones(6))
ops.Gamma[1, 1] = 1.0 

params_adjoint = mso_params_ops(T=T, 
t = 0,
x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
u = 0.0 .* u,
n = zeros(6, T+1), 
q = q_kf,
data_steps = data_steps,
data = zeros(6,T+1),
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

# change the number here to adjust how many steps of grad descent are run
grad_descent(2, params_adjoint)