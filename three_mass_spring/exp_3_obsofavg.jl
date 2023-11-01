function exp_3_obsofavg()

    # Parameter choices 
    T = 10000          # Total number of steps to integrate
    r = 0.5               # spring coefficient 
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))                       # known forcing function 
    q_kf(t) = 0.5 * q_true(t)                                           # forcing seen by KF (and adjoint)
    data_steps1 = [k for k in 2500:300:7000]
    data_steps2 = [k for k in 7300:150:T]
    data_steps = [data_steps1; data_steps2]     # steps where data will be assimilated

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
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

    # pure prediction model 
    _ = create_data(params_pred, ops)

    states_noisy = create_data(params_true, ops)

    params_kf = mso_params(T=T, 
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = 0.0 .* u,
    n = 0.0 .* randn(6, T+1), 
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

    # params_adjoint = mso_params(T=T, 
    # x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # u = 0.0 .* u,
    # n = 0.0 .* randn(6, T+1), 
    # q = q_kf,
    # data_steps = data_steps,
    # data = states_noisy,
    # states = zeros(6, T+1),
    # energy = zeros(3, T+1)
    # )

    # grad_descent(100, params_adjoint, ops)

    # params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # params_adjoint.states .= zeros(6, T+1)

    # _ = run_adjoint(params_adjoint,
    #     ops
    # )

    diag = 1 / ops.Q[1,1]
    Q_inv = diag
    R_inv = ops.R^(-1)
    params_adjoint = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = 0.0 .* u,
        n = 0.001 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops.A,
        B = ops.B,
        Gamma = ops.Gamma,
        E = ops.E,
        Q = ops.Q,
        Q_inv = Q_inv,
        R = ops.R,
        R_inv = R_inv,
        K = ops.K,
        Kc = ops.Kc
    )
    grad_descent(100, params_adjoint, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # creating plots for second experiment  
    # plot of the position of mass one 
    mass_1_pos = plot(params_true.states[1,:],
        label = L"x_1(t)"
    )
    plot!(params_pred.states[1,:],
    label = L"\tilde{x}_1(t, -)"
    )
    plot!(params_kf.states[1,:],
        label = L"\tilde{x}_1(t)",
        ls=:dash
    )
    plot!(params_adjoint.states[1,:],
        label = L"\tilde{x}_1(t, +)",
        ls=:dashdot
    )
    vline!(data_steps, 
    label = "",
    ls=:dot,
    lc=:red,
    lw=0.5
    )
    yaxis!("Position")

    # plot of the velocity of mass one 
    mass_1_vel = plot(params_true.states[4,:],
    label = L"x_4(t)"
    )
    plot!(params_pred.states[4,:],
    label = L"\tilde{x}_4(t, -)"
    )
    plot!(params_kf.states[4,:],
        label = L"\tilde{x}_4(t)",
        ls=:dash
    )
    plot!(params_adjoint.states[4,:],
        label = L"\tilde{x}_4(t, +)",
        ls=:dashdot
    )
    vline!(data_steps, 
    label = "",
    ls=:dot,
    lc=:red,
    lw=0.5
    )
    yaxis!("Velocity")

    # plot of the energy 
    # energy = plot(params_true.energy[3,:], 
    #     label = L"\varepsilon(t)"
    # )
    # plot!(params_pred.energy[3,:],
    # label = L"\tilde{\varepsilon}(t, -)")
    # plot!(params_kf.energy[3,:], 
    #     label = L"\tilde{\varepsilon}(t)",
    #     ls=:dash
    # )
    # plot!(params_adjoint.energy[3,:], 
    #     label = L"\tilde{\varepsilon}(t, +)",
    #     ls=:dashdot
    # )
    # vline!(data_steps, 
    # label = "",
    # ls=:dot,
    # lc=:red,
    # lw=0.5
    # )
    # yaxis!("Energy")

    pos_diffs = plot(abs.(params_true.states[1,:] - params_kf.states[1,:]),
        label = L"|x_1(t) - \tilde{x_1}(t)|"
    )
    plot!(abs.(params_true.states[1,:] - params_adjoint.states[1,:]),
        label = L"|x_1(t) - \tilde{x_1}(t, +)|",
        ls=:dash
    )
    vline!(data_steps, 
    label = "",
    ls=:dot,
    lc=:red,
    lw=0.5
    )
    yaxis!("Position")
    xaxis!("Timestep")

    vel_diffs = plot(abs.(params_true.states[4,:] - params_kf.states[4,:]),
    label = L"|x_4(t) - \tilde{x_4}(t)|"
    )
    plot!(abs.(params_true.states[4,:] - params_adjoint.states[4,:]),
        label = L"|x_4(t) - \tilde{x_4}(t, +)|",
        ls=:dash
    )
    vline!(data_steps, 
    label = "",
    ls=:dot,
    lc=:red,
    lw=0.5
    )
    yaxis!("Velocity")
    xaxis!("Timestep")

    # plot of differences in estimates vs. truth 
    # energy_diffs = plot(abs.(params_true.energy[3,:] - params_kf.energy[3,:]),
    #     label = L"\varepsilon(t) - \tilde{\varepsilon}(t)"
    # )
    # plot!(abs.(params_true.energy[3,:] - params_adjoint.energy[3,:]),
    #     label = L"\varepsilon(t) - \tilde{\varepsilon}(t, +)",
    #     ls=:dash
    # )
    # vline!(data_steps, 
    # label = "",
    # ls=:dot,
    # lc=:red,
    # lw=0.5
    # )
    # yaxis!("Energy")
    # xaxis!("Timestep")

    plot(mass_1_pos, mass_1_vel, pos_diffs, vel_diffs, 
        layout = (4,1), 
        fmt = :png,
        dpi = 300, 
        legend = :outerright
    )

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
