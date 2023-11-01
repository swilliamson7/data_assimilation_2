using Plots, Enzyme, LinearAlgebra, Statistics, Random
using Parameters, UnPack, LaTeXStrings
using Checkpointing, Zygote
using Optim

include("misc_functions.jl")
include("structs.jl")
include("kalman_filter.jl")
include("adjoint.jl")

function fg!(F, G, x)
    # Parameter choices 
    T = 10000          # Total number of steps to integrate
    r = 0.5               # spring coefficient 
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
    q_kf(t) = 0.5 * q_true(t)                                           # forcing seen by KF (and adjoint)
    data_steps1 = [k for k in 2500:300:7000]
    # data_steps2 = [k for k in 7300:150:T]
    data_steps = data_steps1     # steps where data will be assimilated

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
    n = 0.01 .* randn(6, T+1), 
    # n = 0.05 .* randn(6, T+1), 
    q = q_true,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops = build_ops(
        params_true, 
        E = zeros(2,6), 
        R = zeros(2,2),
        K = zeros(6,2)
    )

    # assuming data is an average of positions 1 and 2 and velocities 2 and 3
    ops.E[1,1] = 0.5 
    ops.E[1,2] = 0.5
    ops.E[2,5] = 0.5
    ops.E[2,6] = 0.5

    ops.Q[1,1] = cov(params_true.u[:], corrected=false)
    ops.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(2))

    # assuming random forcing on position of mass one 
    ops.Gamma[1, 1] = 1.0 

    states_noisy = create_data(params_true, ops)

    params_adjoint = mso_params(T=T, 
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = 0.0 .* u,
    n = 0.0 .* randn(6, T+1), 
    q = q_kf,
    data_steps = data_steps,
    data = states_noisy,
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    if G !== nothing 

        adjoint_variables = run_adjoint(params_adjoint, ops)
        G .= compute_control_deriv(adjoint_variables, params_adjoint, ops=ops)

    end
    if F !== nothing
        
        total_cost = 0.0 
        Q_inv = 1/ops.Q[1,1]
        for j in params_adjoint.data_steps 

            total_cost = total_cost + (ops.E * params_adjoint.states[:,j] - ops.E * states_noisy[:,j])' * ops.R^(-1) *
                        (ops.E * params_adjoint.states[:,j] - ops.E * states_noisy[:,j]) + params_adjoint.u[:,j]' * Q_inv * params_adjoint.u[:,j]

            return total_cost

        end

    end


end

res = Optim.optimize(Optim.only_fg!(fg!), zeros(6,10001))

# # Parameter choices 
# T = 10000          # Total number of steps to integrate
# r = 0.5               # spring coefficient 
# q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
# q_kf(t) = 0.5 * q_true(t)                                           # forcing seen by KF (and adjoint)
# data_steps1 = [k for k in 2500:300:7000]
# # data_steps2 = [k for k in 7300:150:T]
# data_steps = data_steps1     # steps where data will be assimilated

# rand_forcing = 0.1 .* randn(T+1)
# u = zeros(6, T+1)
# u[1, :] .= rand_forcing

# params_true = mso_params(T = T,
# x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# u = u,
# n = 0.01 .* randn(6, T+1), 
# # n = 0.05 .* randn(6, T+1), 
# q = q_true,
# data_steps = data_steps,
# data = zeros(1,1),
# states = zeros(6, T+1),
# energy = zeros(3, T+1)
# )


# ops = build_ops(
#     params_true, 
#     E = zeros(2,6), 
#     R = zeros(2,2),
#     K = zeros(6,2)
# )

# # assuming data is an average of positions 1 and 2 and velocities 2 and 3
# ops.E[1,1] = 0.5 
# ops.E[1,2] = 0.5
# ops.E[2,5] = 0.5
# ops.E[2,6] = 0.5

# ops.Q[1,1] = cov(params_true.u[:], corrected=false)
# ops.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(2))

# # assuming random forcing on position of mass one 
# ops.Gamma[1, 1] = 1.0 

# states_noisy = create_data(params_true, ops)



# params_adjoint = mso_params(T=T, 
# x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# u = res.minimizer,
# n = 0.0 .* randn(6, T+1), 
# q = q_kf,
# data_steps = data_steps,
# data = states_noisy,
# states = zeros(6, T+1),
# energy = zeros(3, T+1)
# )

# _ = run_adjoint(params_adjoint, 
#     ops
# )

# mass_1_pos = plot(params_true.states[1,:],
#         label = L"x_1(t)"
#     )
# plot!(params_adjoint.states[1,:],
# label = L"\tilde{x}_1(t, +)",
# ls=:dashdot
# )