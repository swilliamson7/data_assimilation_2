using Plots, Enzyme, LinearAlgebra, Statistics, Random
using Parameters, UnPack

include("misc_functions.jl")
include("structs.jl")
include("kalman_filter.jl")
include("adjoint.jl")

# Chosen random seed 649, will be used for all experiments 
Random.seed!(649)

# Parameter choices 
T = 10000          # Total number of steps to integrate
r = 0.5               # spring coefficient 
q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function 
q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF (and adjoint)
data_steps1 = [k for k in 2000:150:4000]
data_steps2 = [k for k in 4300:300:7500]
data_steps = [data_steps1; data_steps2]        # steps where data will be assimilated

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

ops = build_ops(params_true)

# assuming data of all positions and velocities -> E is the identity operator 
ops.E .= Diagonal(ones(6))

ops.Q[1,1] = cov(params_true.u[:], corrected=false)
ops.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

# assuming random forcing on position of mass one 
ops.Gamma[1, 1] = 1.0 

states_noisy = create_data(params_true, ops)

params_kf = mso_params(T=T, 
x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
u = 0.0 .* u,
n = 0.001 .* randn(6, T+1), 
q = q_kf,
data_steps = data_steps,
data = states_noisy,
states = zeros(6, T+1),
energy = zeros(3, T+1)
)

uncertainty, kinetic_energy_kf, potential_energy_kf, energy_kf = run_kalman_filter(
    params_kf, 
    ops, 
    states_noisy, 
    data_steps = data_steps
)

params_adjoint = mso_params(T=T, 
x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
u = 0.0 .* u,
n = 0.001 .* randn(6, T+1), 
q = q_kf,
data_steps = data_steps,
data = states_noisy,
states = zeros(6, T+1),
energy = zeros(3, T+1)
)

grad_descent(2000, params_adjoint, ops)

params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
params_adjoint.states .= zeros(6, T+1)

adjoint_variables = run_adjoint(params_adjoint, 
    ops, 
    data_steps = params_adjoint.data_steps
)

plot(params_true.states[1,:])
plot!(params_kf.states[1,:], ls=:dash)
plot!(params_adjoint.states[1,:], ls=:dashdot)

#### checking the derivative returned by Enzyme 

# for_checking = 0.0
# for j in data_steps 

#     for_checking = for_checking + (params_adjoint.states[:,j] - states_noisy[:,j])' * ops.R^(-1) *
#                     (params_adjoint.states[:,j] - states_noisy[:,j]
#     )

# end

# steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
# diffs = []
# params_fc = mso_params(T=T, 
# x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# u = 0.0 .* u,
# n = 0.001 .* randn(6, T+1), 
# q = q_KF,
# data_steps = data_steps,
# data = states_noisy,
# states = zeros(6, T+1)
# )
# for s in steps 

#     params_fc.x .= [1.0; 0.0; 0.0; 0.0; 0.0; 0.0] + [s; 0.; 0.; 0.; 0.; 0.]

#     total_cost = 0.0 
#     temp = 0.0 
#     for j = 2:params_fc.T

#         params_fc.x .= ops.A * params_fc.x + ops.B * [params_fc.q(temp); 0.; 0.; 0.; 0.; 0.] 

#         if j in data_steps 
#             total_cost += (params_fc.x - states_noisy[:,j])' * ops.R^(-1) * (params_fc.x - states_noisy[:,j])
#         end

#         temp += params_fc.dt
        
#     end  

#     push!(diffs, (total_cost - for_checking)/s)

# end
