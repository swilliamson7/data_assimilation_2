function exp4_cost_eval(u_guess, params)

    T = params.T
    params.x .= [1.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    params.states .= zeros(6, T+1)
    params.J = 0.0
    params.energy .= zeros(3, T+1)

    params.u .= reshape(u_guess, 6, T+1)

    integrate(params)

    return params.J

end

function exp4_grad_eval(G, u_guess, params)

    T = params.T

    params.x .= [1.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    params.states .= zeros(6, T+1)
    params.J = 0.0
    params.energy .= zeros(3, T+1)

    params.u .= reshape(u_guess, 6, T+1)
    dparams = Enzyme.make_zero(params)
    dparams.J = 1.0
    autodiff(Enzyme.Reverse, integrate, Duplicated(params, dparams))

    G .= vec(dparams.u)

    return nothing

end

function exp4_FG(F, G, u_guess, params_adjoint)

    G === nothing || exp4_grad_eval(G, u_guess, params_adjoint)
    F === nothing || return exp4_cost_eval(u_guess, params_adjoint)

end

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
    n = 0.01 .* randn(6, T+1),
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

    diag = 1 / ops.Q[1,1]
    Q_inv = diag
    R_inv = ops.R^(-1)
    params_adjoint = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        u = 0.0 .* u,
        n = 0.001 .* randn(6, T+1),
        q = q_kf,
        # J = 0.0,
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

    G = vec(params_adjoint.u)

    fg!_closure(F, G, u_guess) = exp4_FG(F, G, u_guess, params_adjoint)
    obj_fg = Optim.only_fg!(fg!_closure)
    result = Optim.optimize(obj_fg, vec(params_adjoint.u), Optim.LBFGS(), Optim.Options(show_trace=true,iterations=10))

    T = params_adjoint.T

    params_adjoint.x .= [1.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.states .= zeros(6, T+1)
    params_adjoint.J = 0.0
    params_adjoint.energy .= zeros(3, T+1)

    params_adjoint.u .= reshape(result.minimizer, 6, T+1)
    integrate(params_adjoint)

    std_position = zeros(T+1)
    std_velocity = zeros(T+1)
    for j = 1:T+1 
        std_position[j] = sqrt(uncertainty[j][2,2])
        std_velocity[j] = sqrt(uncertainty[j][5,5])
    end

    # plot of fixed displacement
    fig = Figure();
    ax1 = Axis(fig[1,1], ylabel="Position");
    lines!(ax1, params_true.states[2,:], label = L"x_2(t)");
    lines!(ax1, params_pred.states[2,:], label=L"x_2(t, -)")
    lines!(ax1, params_kf.states[2,:], linestyle=:dash, label = L"\tilde{x}_2(t)");
    band!(ax1, 1:10001, params_kf.states[2,:]-std_position, params_kf.states[2,:]+std_position, color=(:lightgreen,0.5));
    lines!(ax1, params_adjoint.states[2,:], label = L"\tilde{x}_2(t, +)", linestyle=:dashdot);
    vlines!(ax1, data_steps, color=:gray75, linestyle=:dot);

    ax2 = Axis(fig[2,1], ylabel="Velocity");
    lines!(ax2, params_true.states[5,:],label = L"x_5(t)");
    lines!(ax2, params_pred.states[5,:], label=L"x_5(t, -)")
    lines!(ax2, params_kf.states[5,:], linestyle=:dash, label=L"\tilde{x}_5(t)");
    band!(ax2, 1:10001, params_kf.states[5,:]-std_velocity, params_kf.states[5,:]+std_velocity, color=(:lightgreen,0.5));
    lines!(ax2, params_adjoint.states[5,:], linestyle=:dashdot,label = L"\tilde{x}_5(t, +)");
    vlines!(ax2, data_steps, color=:gray75, linestyle=:dot);

    # plot of energy 
    ax3 = Axis(fig[3,1], ylabel="Energy")
    lines!(ax3, params_true.energy[3,:],label=L"\varepsilon(t)")
    lines!(ax3, params_pred.energy[3,:], label=L"\tilde{\varepsilon}(t,-)")
    lines!(ax3, params_kf.energy[3,:], linestyle=:dash, label=L"\tilde{\varepsilon}(t)")
    lines!(ax3, params_adjoint.energy[3,:], linestyle=:dashdot, label=L"\tilde{\varepsilon}(t,+)")
    vlines!(ax3, data_steps, color=:gray75, linestyle=:dot);

    # plot of differences
    ax4 = Axis(fig[4,1], ylabel="Position", xlabel="Timestep")
    lines!(ax4, abs.(params_true.states[2,:] - params_kf.states[2,:]),label = L"|x_2(t) - \tilde{x}_2(t)|")
    lines!(ax4, abs.(params_true.states[2,:] - params_adjoint.states[2,:]),label = L"|x_2(t) - \tilde{x}_2(t, +)|")
    vlines!(ax4, data_steps, color=:gray75, linestyle=:dot);

    fig[1,2] = Legend(fig, ax1)
    fig[2,2] = Legend(fig, ax2)
    fig[3,2] = Legend(fig, ax3)
    fig[4,2] = Legend(fig, ax4)

    return params_true, params_pred, params_kf, params_adjoint, fig

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
