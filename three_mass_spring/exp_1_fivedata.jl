function exp1_cost_eval(u_guess, params)

    T = params.T
    params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params.states .= zeros(6, T+1)
    params.J = 0.0
    params.energy .= zeros(3, T+1)

    params.u .= reshape(u_guess, 6, T+1)

    integrate(params)

    return params.J

end

function exp1_grad_eval(G, u_guess, params)

    T = params.T

    params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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

function exp1_FG(F, G, u_guess, params_adjoint)

    G === nothing || exp1_grad_eval(G, u_guess, params_adjoint)
    F === nothing || return exp1_cost_eval(u_guess, params_adjoint)

end

function exp_1_fivedata()

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

    # pure prediction model 
    _ = create_data(params_pred, ops)

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

    uncertainty = run_kalman_filter(
        params_kf, 
        ops
    )

    # with Enzyme
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

    G = vec(params_adjoint.u)

    fg!_closure(F, G, u_guess) = exp1_FG(F, G, u_guess, params_adjoint)
    obj_fg = Optim.only_fg!(fg!_closure)
    result = Optim.optimize(obj_fg, vec(params_adjoint.u), Optim.LBFGS(), Optim.Options(show_trace=true,iterations=7))

    T = params_adjoint.T

    params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.states .= zeros(6, T+1)
    params_adjoint.J = 0.0
    params_adjoint.energy .= zeros(3, T+1)

    params_adjoint.u .= reshape(result.minimizer, 6, T+1)
    integrate(params_adjoint)

    # dparams_adjoint = grad_descent(100, params_adjoint, [1.,0.,0.,0.,0.,0.])

    ## plots for exp 1

    # plot of the position of mass one
    fig = Figure();
    ax = Axis(fig[1,1], ylabel="Position");
    lines!(ax, params_true.states[1,:], label=L"x_1(t)");
    lines!(ax, params_pred.states[1,:], label=L"\tilde{x}(t, -)");
    lines!(ax, params_kf.states[1,:], label=L"\tilde{x}_1(t)", linestyle=:dash);
    lines!(ax, params_adjoint.states[1,:], label=L"\tilde{x}_1(t, +)", linestyle=:dashdot);
    vlines!(ax, data_steps, color=:gray75, linestyle=:dot);
    
    # plot of the energy 
    ax1 = Axis(fig[2,1], ylabel="Energy");
    lines!(ax1, params_true.energy[3,:], label=L"\varepsilon(t)");
    lines!(ax1, params_pred.energy[3,:], label = L"\tilde{\varepsilon}(t, -)");
    lines!(ax1, params_kf.energy[3,:], label = L"\tilde{\varepsilon}(t)",linestyle=:dash);
    lines!(ax1, params_adjoint.energy[3,:], label = L"\tilde{\varepsilon}(t, +)",linestyle=:dashdot);
    vlines!(ax1, data_steps, color=:gray75, linestyle=:dot);

    # plot of differences in estimates vs. truth
    ax2 = Axis(fig[3,1], ylabel="Energy", xlabel="Timestep");
    lines!(ax2,abs.(params_true.energy[3,:] - params_kf.energy[3,:]),label = L"|\varepsilon(t) - \tilde{\varepsilon}(t)|");
    lines!(ax2, abs.(params_true.energy[3,:] - params_adjoint.energy[3,:]),label = L"|\varepsilon(t) - \tilde{\varepsilon}(t, +)|",linestyle=:dash);
    vlines!(ax2, data_steps, color=:gray75, linestyle=:dot);

    fig[1,2] = Legend(fig, ax)
    fig[2,2] = Legend(fig, ax1)
    fig[3,2] = Legend(fig, ax2)

    return params_true, params_pred, params_kf, params_adjoint, fig
end

#### checking the derivative returned by Enzyme 

# for_checking = 0.0
# for j in data_steps 

#     for_checking = for_checking + (params_adjoint.states[:,j] - states_noisy[:,j])' * R_inv *
#                     (params_adjoint.states[:,j] - states_noisy[:,j]) + params_adjoint.u[:,j]' * Q_inv * params_adjoint.u[:,j]
                   

# end

# steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
# diffs = []
# params_fc = mso_params(T=T, 
# x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
# u = 0.0 .* u,
# n = 0.001 .* randn(6, T+1), 
# q = q_kf,
# data_steps = data_steps,
# data = states_noisy,
# states = zeros(6, T+1),
# energy = zeros(3, T+1)
# )
# for s in steps

#     params_fc.x .= [1.0;0.0;0.0;0.0;0.0;0.0]
#     params_fc.u[:,2499] .= zeros(6,1)
#     params_fc.u[:,2499] = params_fc.u[:,2499] + [s; 0.; 0.; 0.; 0.; 0.]

#     total_cost = 0.0
#     temp = 0.0
#     for j = 2:T+1

#         params_fc.x .= ops.A * params_fc.x + ops.B * [params_fc.q(temp); 0.; 0.; 0.; 0.; 0.] + ops.Gamma * params_fc.u[:, j-1]


#         if j in data_steps 
#             total_cost = total_cost + (params_fc.x - states_noisy[:,j])' * ops.R^(-1) * (params_fc.x - states_noisy[:,j]) 
#             + params_fc.u[:,j]' * (1/ops.Q[1,1]) * params_fc.u[:,j]
#         end

#         temp += params_fc.dt

#     end

#     @show total_cost

#     push!(diffs, (total_cost - for_checking)/s)

# end
