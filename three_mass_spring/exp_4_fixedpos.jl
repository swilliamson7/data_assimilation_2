# using Plots, Enzyme, LinearAlgebra, Statistics, Random
# using Parameters, UnPack

# include("misc_functions.jl")
# include("structs.jl")
# include("kalman_filter.jl")
# include("adjoint.jl")

# # Chosen random seed 649, will be used for all experiments 
# Random.seed!(649)

function create_data1(params, ops)

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

        x[2] = 3.0
        x[5] = 0.0

        states[:, j] .= copy(x)

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,j] = [kin;ptl;total]

        states_noisy[:, j] = copy(x) + n[:, j]
        temp += dt

    end 

    return states_noisy

end

function exp_4_fixedpos(;optm_steps = 100)

    # Parameter choices 
    T = 10000          # Total number of steps to integrate
    r = 0.5               # spring coefficient 
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function 
    q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    data_steps1 = [k for k in 3000:200:7000]         # steps where data will be assimilated
    # data_steps2 = [k for k in 7200:100:9000]
    # data_steps = [data_steps1;data_steps2]
    data_steps = data_steps1
    # data_steps = [t for t in 1:T]


    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = mso_params(T = T,
    x = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
    n = 0.05 .* randn(6, T+1), 
    q = q_true,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    params_pred = mso_params(T = T,
    x = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
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

    # pure prediction model 
    _ = create_data(params_pred, ops)

    states_noisy = create_data1(params_true, ops)

    params_kf = mso_params(T=T, 
    x = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    u = 0.0 .* u,
    n = zeros(1,1), 
    q = q_kf,
    data_steps = data_steps,
    data = states_noisy,
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    uncertainty = run_kalman_filter(
        params_kf,
        ops
    )

    params_adjoint = mso_params(T=T, 
    x = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    u = 0.0 .* u,
    n = zeros(1,1), 
    q = q_kf,
    data_steps = data_steps,
    data = states_noisy,
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    grad_descent(optm_steps, params_adjoint, ops)

    params_adjoint.x .= [1.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.states .= zeros(6, T+1)

    _ = run_adjoint(params_adjoint, 
        ops
    )

    # params_adjoint2 = mso_params_ops(T=T, 
    # t = 0,
    # x = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    # u = 0.0 .* u,
    # n = 0.0 .* randn(6, T+1), 
    # q = q_kf,
    # data_steps = data_steps,
    # data = states_noisy,
    # states = zeros(6, T+1),
    # energy = zeros(3, T+1), 
    # A = ops.A,
    # B = ops.B, 
    # Gamma = ops.Gamma, 
    # E = ops.E, 
    # Q = ops.Q, 
    # R = ops.R,
    # K = ops.K,
    # Kc = ops.Kc
    # )

    # grad_descent(100, params_adjoint, params_adjoint.x)

    # plot of fixed displacement 
    fixed_pos = plot(params_true.states[2,:],
    label = L"x_2(t)"
    )
    # plot!(params_pred.states[2, :],
    # label = L"\tilde{x}_2(t, -)"
    # )
    # for uncertainty ribbon 
    to_plot1 = []
    to_plot2 = []
    for j = 1:T+1 
        push!(to_plot1, sqrt(uncertainty[j][2,2]))
        push!(to_plot2, sqrt(uncertainty[j][5,5]))
    end
    plot!(params_kf.states[2,:], ribbon = to_plot1, ls=:dash, label = L"\tilde{x}_2(t)")
    
    plot!(params_adjoint.states[2,:], ls=:dashdot,
    label = L"\tilde{x}_2(t, +)")
    vline!(data_steps, 
    label = "",
    ls=:dot,
    lc=:red,
    lw=0.5
    )
    ylabel!("Position")

    # plot of fixed velocity 
    fixed_vel = plot(params_true.states[5,:],
    label = L"x_5(t)")
    # plot!(params_pred.states[5, :],
    # label = L"\tilde{x}_5(t, -)"
    # )
    plot!(params_kf.states[5,:], ls=:dash, label = L"\tilde{x}_5(t)", ribbon = to_plot2)
    plot!(params_adjoint.states[5,:], ls=:dashdot,
    label = L"\tilde{x}_5(t, +)")
    vline!(data_steps, 
    label = "",
    ls=:dot,
    lc=:red,
    lw=0.5
    )
    ylabel!("Velocity")

    # plot of energy 
    energy = plot(params_true.energy[3,:],
    label=L"\varepsilon(t)")
    plot!(params_pred.energy[3,:], label=L"\tilde{\varepsilon}(t,-)")
    plot!(params_kf.energy[3,:], ls=:dash, label=L"\tilde{\varepsilon}(t)")
    plot!(params_adjoint.energy[3,:], ls=:dashdot, label=L"\tilde{\varepsilon}(t,+)")
    vline!(data_steps, 
    label = "",
    ls=:dot,
    lc=:red,
    lw=0.5
    )
    ylabel!("Energy")

    # plot of differences 
    diffs = plot(abs.(params_true.states[2,:] - params_kf.states[2,:]),
    label = L"|x_2(t) - \tilde{x}_2(t)|"
    )
    plot!(abs.(params_true.states[2, :] - params_adjoint.states[2,:]),
    label = L"|x_2(t) - \tilde{x}_2(t, +)|"
    )
    ylabel!("Position")
    xlabel!("Timestep")

    plot(fixed_pos, fixed_vel, energy, diffs, 
    layout = (4,1), 
    fmt = :png,
    dpi = 300, 
    legend = :outerright)
    

end

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
