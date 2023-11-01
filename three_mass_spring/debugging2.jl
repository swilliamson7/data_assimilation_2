using Enzyme
using UnPack

mutable struct mso_params_ops{F<:Function}

    T::Int            
    t::Int           
    dt::Float64 

    x::Vector{Float64}
    u::Matrix{Float64} 
    n::Matrix{Float64}  

    k::Int          
    r::Float64   

    q::F

    J::Float64

    data_steps::Vector{Int64}     
    data::Matrix{Float64}

    states::Matrix{Float64}
    energy::Matrix{Float64}  

    A::Matrix{Float64}  
    B::Matrix{Float64}
    Gamma::Matrix{Float64} 

    E::Matrix{Float64}   

    Q::Matrix{Float64}
    R::Matrix{Float64}

    K::Matrix{Float64}    
    Kc::Matrix{Float64}  

end

# integrate function with UnPack
# function integrate(mso_struct)

#     @unpack A, B, Gamma, E, R, Kc, Q = mso_struct
#     @unpack T, x, u, dt, states, data, energy, q =  mso_struct
#     @unpack data_steps, J = mso_struct 

#     Q_inv = 1/Q[1,1]

#     # run the model forward to get estimates for the states 
#     temp = 0.0 
#     for t = 2:T+1 

#         x .= A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t]
#         states[:, t] .= copy(x) 

#         temp += dt

#         if t in data_steps 

#             # mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R^(-1) * (E * x - E * data[:, t]) 

#             if t != T && t in data_steps 
#                 mso_struct.J = mso_struct.J + u[:, t]' * Q_inv * u[:, t]
#             end

#         end

#     end 

#     return nothing 

# end

# without UnPack
function integrate(mso_struct)

    Q_inv = 1/mso_struct.Q[1,1]

    # run the model forward to get estimates for the states 
    temp = 0.0 
    for t = 2:mso_struct.T+1 

        mso_struct.x .= mso_struct.A * mso_struct.x + mso_struct.B * [mso_struct.q(temp); 0.; 0.; 0.; 0.; 0.] 
        + mso_struct.Gamma * mso_struct.u[:, t]
        mso_struct.states[:, t] .= copy(mso_struct.x) 

        temp += mso_struct.dt

        if t in mso_struct.data_steps 

            if t != mso_struct.T && t in mso_struct.data_steps 
                mso_struct.J = mso_struct.J + mso_struct.u[:, t]' * Q_inv * mso_struct.u[:, t]
            end

        end

    end 

    return nothing 

end

# Parameter choices 
T = 10000          
r = 0.5
q(t) = 0.1 * cos(2 * pi * t / (2.5 / r))            
data_steps = [1500 + k*1000 for k in 1:5]     
dt = 0.001
k = 30

rand_forcing = 0.1 .* ones(T+1)
u = zeros(6, T+1)
u[1, :] .= rand_forcing
n = 0.05 .* ones(6, T+1)

Q = zeros(6,6)
K = zeros(6,6)
Gamma = zeros(6,6)
for_building = [1. 0. 0.
                0. 1. 0.
                0. 0. 1.]
Rc = -r .* for_building
Kc = [-2.0*k k 0.0; k -3.0*k k; 0.0 k -2.0*k]
Ac = [zeros(3,3) for_building
     Kc Rc
]

A = [1. 0. 0. 0. 0. 0.
    0. 1. 0. 0. 0. 0.
    0. 0. 1. 0. 0. 0.
    0. 0. 0. 1. 0. 0.
    0. 0. 0. 0. 1. 0.
    0. 0. 0. 0. 0. 1.]  + dt .* Ac

Q[1,1] = 0.2
R = 0.1 .* [1. 0. 0. 0. 0. 0.
            0. 1. 0. 0. 0. 0.
            0. 0. 1. 0. 0. 0.
            0. 0. 0. 1. 0. 0.
            0. 0. 0. 0. 1. 0.
            0. 0. 0. 0. 0. 1.] 

# assuming random forcing on position of mass one 
Gamma[1, 1] = 1.0 

# create data from the true setup
states_noisy = zeros(6, T+1)

B = [1. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.]

Gamma = B 

E = [1. 0. 0. 0. 0. 0.
     0. 1. 0. 0. 0. 0.
     0. 0. 1. 0. 0. 0.
     0. 0. 0. 1. 0. 0.
     0. 0. 0. 0. 1. 0.
     0. 0. 0. 0. 0. 1.] 

params_adjoint = mso_params_ops(T, 
    0,
    .001,
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    0.0 .* u,
    0.001 .* ones(6, T+1), 
    30, 
    0.5, 
    q,
    0.0,
    data_steps,
    states_noisy,
    zeros(6, T+1),
    zeros(3, T+1), 
    A,
    B, 
    Gamma, 
    E, 
    Q, 
    R,
    K,
    Kc
)

# integrate(params_adjoint)

dparams = deepcopy(params_adjoint)
autodiff(Reverse, integrate, Duplicated(params_adjoint, dparams))

# snaps = 1
# verbose = 1
# revolve = Revolve{mso_params_ops}(params_adjoint.T, snaps; verbose=verbose)

# dparams = Zygote.gradient(
#     cp_adjoint, 
#     params_adjoint,
#     q,
#     revolve
# )
