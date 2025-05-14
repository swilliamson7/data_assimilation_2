function exp6_integrate(mso_struct::mso_params_ops)

    @unpack B, Gamma, E, R, R_inv, Kc, Q, Q_inv = mso_struct
    @unpack r = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct

    k = mso_struct.k

    states[:,1] .= x

    kin, ptl, total = compute_energy(states[:,1], Kc)
    energy[:,1] = [kin;ptl;total]

    Ac = zeros(6,6)
    Ac[1,4] = 1
    Ac[2,5] = 1
    Ac[3,6] = 1
    Ac[4,1] = -2*k
    Ac[4,2] = k
    Ac[4,4] = -r
    Ac[5,1] = k
    Ac[5,2] = -3*k
    Ac[5,3] = k
    Ac[5,5] = -r
    Ac[6,2] = k
    Ac[6,3] = -2*k
    Ac[6,6] = -r
    A = diagm(ones(6)) + dt .* Ac

    # run the model forward to get estimates for the states 
    temp = 0.0
    for t = 2:T+1

        x[:] = A * x + B * [q(temp); 0.; 0.; 0.; 0.; 0.] + Gamma * u[:, t-1]
        states[:, t] .= copy(x)

        temp += dt

        if t in data_steps

            mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * R_inv * (E * x - E * data[:, t])
            + u[:, t]' * Q_inv * u[:, t]


        end

        kin, ptl, total = compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

    end

    return nothing

end

function exp6_cost_eval(uk_guess, params_adjoint)

    T = params_adjoint.T
    params_adjoint.k = uk_guess[end]
    params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.u .= reshape(uk_guess[1:end-1], 6, T+1)
    params_adjoint.states .= zeros(6, T+1)
    params_adjoint.J = 0.0
    params_adjoint.energy .= zeros(3, T+1)

    exp6_integrate(params_adjoint)

    return params_adjoint.J

end

function exp6_gradient_eval(G, uk_guess, params_adjoint)

    T = params_adjoint.T

    params_adjoint.k = uk_guess[end]
    params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.u = reshape(uk_guess[1:end-1], 6, T+1)
    params_adjoint.states .= zeros(6, T+1)
    params_adjoint.J = 0.0
    params_adjoint.energy .= zeros(3, T+1)

    dparams = Enzyme.Compiler.make_zero(params_adjoint)
    dparams.J = 1.0

    autodiff(Enzyme.Reverse, exp6_integrate, Duplicated(params_adjoint, dparams))

    G[1:end-1] = vec(dparams.u)
    G[end] = dparams.k

end

function exp6_FG(F, G, uk_guess, params_adjoint)

    G === nothing || exp6_gradient_eval(G, uk_guess, params_adjoint)
    F === nothing || return exp6_cost_eval(uk_guess, params_adjoint)

end

function G_param_forcing(G, k_guess, params)

    T = params.T

    params.k = k_guess[1]
    params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params.states .= zeros(6, T+1)
    params.J = 0.0
    params.energy .= zeros(3, T+1)

    dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.

    autodiff(Reverse, integrate2, Duplicated(params, dparams))

    G[1] = dparams.k

end

function FG_param_forcing(F, G, k_guess, params)

    G === nothing || G_param_forcing(G, k_guess, params)
    F === nothing || return F_param_forcing(k_guess, params)

end

function exp_6_param_forcing(;optm_steps = 100, k_guess=35.)

    # Parameter choices
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    # q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [k for k in 375:375:10000]         # steps where data will be assimilated
    # data_steps2 = [k for k in 7200:100:9000]
    # data_steps = [data_steps1;data_steps2]
    data_steps = data_steps1
    # data_steps = [t for t in 1:T]

    sigmaf = 0.1            # standard deviation of random forcing
    sigmad = 0.05           # standard deviation of noise added to data


    rand_forcing = sigmaf .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = mso_params(T = T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        k = 30,
        n = sigmad .* randn(6, T+1),
        q = q_true,
        data_steps = data_steps,
        data = zeros(1,1),
        states = zeros(6, T+1),
        energy = zeros(3, T+1)
    )

    params_pred = mso_params(T = T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        q = q_kf,
        k = k_guess,
        data_steps = data_steps,
        data = zeros(1,1),
        states = zeros(6, T+1),
        energy = zeros(3, T+1)
    )

    ops_true = build_ops(params_true)
    ops_pred = build_ops(params_pred)

    # assuming data of all positions and velocities -> E is the identity operator
    ops_true.E .= Diagonal(ones(6))
    ops_pred.E .= Diagonal(ones(6))

    ops_true.Q[1,1] = cov(params_true.u[:], corrected=false)
    ops_pred.Q[1,1] = cov(params_true.u[:], corrected=false)

    ops_true.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))
    ops_pred.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops_true.Gamma[1,1] = 1.0
    ops_pred.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = create_data(params_pred, ops_pred)

    states_noisy = create_data(params_true, ops_true)

    diag = 1 / ops_pred.Q[1,1]
    Q_inv = diag

    ###############################
    R_inv = ops_pred.R^(-1)
    # R_inv = ops_pred.E
    ###############################

    params_kf = mso_params(T=T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = 0.0 .* u,
        k = k_guess,
        n = zeros(1,1),
        q = q_kf,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1)
    )

    uncertainty = run_kalman_filter(
        params_kf,
        ops_pred
    )

    params_adjoint = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = 0.0 .* u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k_guess,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops_pred.A,
        B = ops_pred.B,
        Gamma = ops_pred.Gamma,
        E = ops_pred.E,
        Q = 0.0 .* ops_pred.Q,
        Q_inv = Q_inv,
        R = ops_pred.R,
        R_inv = R_inv,
        K = ops_pred.K,
        Kc = ops_pred.Kc
    )

    fg!_closure(F, G, uk) = exp6_FG(F, G, uk, params_adjoint)
    obj_fg = Optim.only_fg!(fg!_closure)
    result = Optim.optimize(obj_fg, [vec(params_adjoint.u); k_guess], Optim.LBFGS(), Optim.Options(show_trace=true, iterations=10))

    T = params_adjoint.T

    params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.states .= zeros(6, T+1)
    params_adjoint.J = 0.0
    params_adjoint.energy .= zeros(3, T+1)
    params_adjoint.k = result.minimizer[end]
    params_adjoint.u = reshape(result.minimizer[1:6*(T+1)], 6, T+1)
    
    # build a new operator with the optimized k
    Rc = -params_adjoint.r .* diagm(ones(3))
    Kc = [-2*params_adjoint.k params_adjoint.k 0; params_adjoint.k -3*params_adjoint.k params_adjoint.k; 0 params_adjoint.k -2*params_adjoint.k]
    Ac = [zeros(3,3) diagm(ones(3))
         Kc Rc
    ]
    A = diagm(ones(6)) + params_adjoint.dt .* Ac
    params_adjoint.A = A

    exp6_integrate(params_adjoint)

    fig = Figure();

    # position
    ax1 = Axis(fig[1,1], ylabel="Position");
    vlines!(ax1, data_steps, color=:gray75, linestyle=:dot);
    lines!(ax1, params_true.states[1,:], label=L"x_1(t)");
    lines!(ax1, params_pred.states[1,:], label=L"\tilde{x}_1(t, -)");
    lines!(ax1, params_kf.states[1,:], label=L"\tilde{x}_1(t)", linestyle=:dash);
    lines!(ax1, params_adjoint.states[1,:], label=L"\tilde{x}_1(t, +)", linestyle=:dashdot);

    ax2 = Axis(fig[2,1], ylabel="Velocity");
    vlines!(ax2, data_steps, color=:gray75, linestyle=:dot);
    lines!(ax2, params_true.states[4,:], label=L"x_4(t)");
    lines!(ax2, params_pred.states[4,:], label=L"\tilde{x}_4(t, -)");
    lines!(ax2, params_kf.states[4,:], label=L"\tilde{x}_4(t)", linestyle=:dash);
    lines!(ax2, params_adjoint.states[4,:], label=L"\tilde{x}_4(t, +)", linestyle=:dashdot);

    # plot of the energy
    ax3 = Axis(fig[3,1], ylabel="Energy");
    vlines!(ax3, data_steps, color=:gray75, linestyle=:dot);
    lines!(ax3, params_true.energy[3,:], label=L"\varepsilon(t)");
    lines!(ax3, params_pred.energy[3,:], label = L"\tilde{\varepsilon}(t, -)");
    lines!(ax3, params_kf.energy[3,:], label = L"\tilde{\varepsilon}(t)",linestyle=:dash);
    lines!(ax3, params_adjoint.energy[3,:], label = L"\tilde{\varepsilon}(t, +)",linestyle=:dashdot);

    # plot of differences in estimates vs. truth
    ax4 = Axis(fig[4,1], ylabel="Energy", xlabel="Timestep");
    vlines!(ax4, data_steps, color=:gray75, linestyle=:dot);
    lines!(ax4,abs.(params_true.energy[3,:] - params_kf.energy[3,:]),label = L"|\varepsilon(t) - \tilde{\varepsilon}(t)|");
    lines!(ax4, abs.(params_true.energy[3,:] - params_adjoint.energy[3,:]),label = L"|\varepsilon(t) - \tilde{\varepsilon}(t, +)|",linestyle=:dash);

    fig[1,2] = Legend(fig, ax1)
    fig[2,2] = Legend(fig, ax2)
    fig[3,2] = Legend(fig, ax3)
    fig[4,2] = Legend(fig, ax4)

    return params_true, params_pred, params_kf, params_adjoint, fig

end

### checking the derivative returned by Enzyme, not edited yet

function enzyme_check_param_forcing(;k_guess=20.)

    # Parameter choices
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    # q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [k for k in 3000:200:7000]         # steps where data will be assimilated
    # data_steps2 = [k for k in 7200:100:9000]
    # data_steps = [data_steps1;data_steps2]
    data_steps = data_steps1
    # data_steps = [t for t in 1:T]

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = mso_params(T = T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        k = 30,
        n = 0.0001 .* randn(6, T+1),
        q = q_true,
        data_steps = data_steps,
        data = zeros(1,1),
        states = zeros(6, T+1),
        energy = zeros(3, T+1)
    )

    params_pred = mso_params(T = T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        q = q_kf,
        u = u,
        k = k_guess,
        data_steps = data_steps,
        data = zeros(1,1),
        states = zeros(6, T+1),
        energy = zeros(3, T+1)
    )

    ops_true = build_ops(params_true)
    ops_pred = build_ops(params_pred)

    # assuming data of all positions and velocities -> E is the identity operator
    ops_true.E .= Diagonal(ones(6))
    ops_pred.E .= Diagonal(ones(6))

    ops_true.Q[1,1] = cov(params_true.u[:], corrected=false)
    ops_pred.Q[1,1] = cov(params_true.u[:], corrected=false)

    ops_true.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))
    ops_pred.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops_true.Gamma[1,1] = 1.0
    ops_pred.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = create_data(params_pred, ops_pred)

    states_noisy = create_data(params_true, ops_true)

    diag = 0.0
    Q_inv = diag

    ###############################
    R_inv = ops_pred.R^(-1)
    # R_inv = ops_pred.E
    ###############################

    params_adjoint2 = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k_guess,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops_pred.A,
        B = ops_pred.B,
        Gamma = ops_pred.Gamma,
        E = ops_pred.E,
        Q = 0.0 .* ops_pred.Q,
        Q_inv = Q_inv,
        R = ops_pred.R,
        R_inv = R_inv,
        K = ops_pred.K,
        Kc = ops_pred.Kc
    )

    # params_adjoint2, _, _, _, _ = setup_model(k_guess = [k_guess])
    # T = params_adjoint2.T

    dparams_adjoint2 = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint2), IdDict(), params_adjoint2)
    dparams_adjoint2.J = 1.0
    dparams_adjoint2.k = 0.
    dparams_adjoint2.r = 0.
    dparams_adjoint2.dt = 0.
    dparams_adjoint2.Q_inv = 0.

    autodiff(Reverse, integrate1, Duplicated(params_adjoint2, dparams_adjoint2))

    params_adjoint = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k_guess,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops_pred.A,
        B = ops_pred.B,
        Gamma = ops_pred.Gamma,
        E = ops_pred.E,
        Q = 0.0 .* ops_pred.Q,
        Q_inv = Q_inv,
        R = ops_pred.R,
        R_inv = R_inv,
        K = ops_pred.K,
        Kc = ops_pred.Kc
    )

    integrate1(params_adjoint)

    for_checking = 0.0
    for j in data_steps

        for_checking = for_checking + (params_adjoint.states[:,j] - states_noisy[:,j])' * params_adjoint.R_inv * (params_adjoint.states[:,j] - states_noisy[:,j])

    end

    steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    diffs = []
    params_fc = mso_params_ops(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k_guess,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = ops_pred.A,
        B = ops_pred.B,
        Gamma = ops_pred.Gamma,
        E = ops_pred.E,
        Q = 0.0 .* ops_pred.Q,
        Q_inv = Q_inv,
        R = ops_pred.R,
        R_inv = R_inv,
        K = ops_pred.K,
        Kc = ops_pred.Kc
    )

    for s in steps

        params_fc.x .= [1.0;0.0;0.0;0.0;0.0;0.0]
        params_fc.k = k_guess + s
        r = params_fc.r

        Ac = zeros(6,6)
        Ac[1,4] = 1
        Ac[2,5] = 1
        Ac[3,6] = 1
        Ac[4,1] = -2*params_fc.k
        Ac[4,2] = params_fc.k
        Ac[4,4] = -r
        Ac[5,1] = params_fc.k
        Ac[5,2] = -3*params_fc.k
        Ac[5,3] = params_fc.k
        Ac[5,5] = -r
        Ac[6,2] = params_fc.k
        Ac[6,3] = -2*params_fc.k
        Ac[6,6] = -r
        params_fc.A = diagm(ones(6)) + params_fc.dt .* Ac

        total_cost = 0.0
        temp = 0.0
        for j = 2:T+1

            params_fc.x .= params_fc.A * params_fc.x + params_fc.B * [params_fc.q(temp); 0.; 0.; 0.; 0.; 0.] + params_fc.Gamma * params_fc.u[:, j-1]

            if j in data_steps
                total_cost = total_cost + (params_fc.x - states_noisy[:,j])' * params_fc.R_inv * (params_fc.x - states_noisy[:,j])
                    + u[:, t]' * Q_inv * u[:, t]
            end

            temp += params_fc.dt

        end

        @show total_cost

        push!(diffs, (total_cost - for_checking)/s)

    end

    @show diffs
    @show dparams_adjoint2.k


end


# function grad_descent2(M, params::mso_params_ops)

#     T = params.T

#     u_new = zeros(6, T+1)
#     k_new = 0.0

#     params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     params.states .= zeros(6, T+1)
#     params.energy .= zeros(3, T+1)
#     params.J = 0.0

#     dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
#     dparams.J = 1.0
#     dparams.k = 0.
#     dparams.r = 0.
#     dparams.dt = 0.
#     dparams.Q_inv = 0.

#     print("Beginning grad descent\n")

#     autodiff(Reverse, integrate2, Duplicated(params, dparams))

#     if M == 0
#         return
#     end

#     k_new = params.k - (1 / norm(dparams.k)) * dparams.k

#     for t = 1:T 

#         u_new[:, t] = params.u[:, t] - 1 / (norm(dparams.u[1,:])) * dparams.u[:, t]

#     end

#     j = 1

#     k_old = copy(params.k)
#     k_grad_old = copy(dparams.k)
#     params.k = k_new

#     u_old = copy(params.u)
#     u_grad_old = copy(dparams.u)
#     params.u .= u_new 

#     while norm(u_grad_old) > 500

#         params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#         params.states .= zeros(6, T+1)
#         params.k = k_new
#         params.u .= u_new
#         params.J = 0.0
#         params.energy .= zeros(3, T+1)

#         dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
#         dparams.J = 1.0
#         dparams.k = 0.
#         dparams.r = 0.
#         dparams.dt = 0.
#         dparams.Q_inv = 0.

#         autodiff(Reverse, integrate2, Duplicated(params, dparams))

#         print("Norm of the derivative\n") 
#         @show norm(dparams.k)
#         @show norm(dparams.u[1,:])

#         print("Current guess for k\n")
#         @show params.k

#         print("Objective value\n")
#         @show params.J

#         gammau = 0.0
#         gammak = 0.0

#         numk = 0.0
#         denk = 0.0
#         numu = 0.0
#         denu = 0.0

#         numk = sum(dot(params.k - k_old, dparams.k - k_grad_old))
#         denk = norm(dparams.k - k_grad_old)^2

#         for j = 1:T 

#             numu += sum(dot(params.u[:,j] - u_old[:,j], dparams.u[:,j] - u_grad_old[:,j]))
#             denu += norm(dparams.u[:,j] - u_grad_old[:,j])^2

#         end

#         gammak = (abs(numk) / denk)
#         gammau = (abs(numu) / denu)

#         k_new = params.k - gammak * dparams.k
#         for t = 1:T
#             u_new[:, t] = params.u[:, t] - gammau * dparams.u[:, t]
#         end

#         k_old = copy(params.k)
#         k_grad_old = copy(dparams.k)
#         params.k = k_new
#         dparams.k = 0.0

#         u_old = copy(params.u)
#         u_grad_old = copy(dparams.u)
#         params.u .= u_new
#         dparams.u = zeros(6, T+1)

#         j += 1

#         if j > M
#             break
#         end

#     end

#     return nothing

# end

# function F_param_forcing(k_guess, params)

#     T = params.T
#     params.k = k_guess[1]
#     params.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     params.states .= zeros(6, T+1)
#     params.J = 0.0
#     params.energy .= zeros(3, T+1)

#     integrate2(params)

#     return params.J

# end