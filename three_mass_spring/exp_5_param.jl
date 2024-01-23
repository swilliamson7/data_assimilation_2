function integrate1(mso_struct::mso_params_ops)

    @unpack B, Gamma, E, R, R_inv, Kc, Q, Q_inv = mso_struct
    @unpack r = mso_struct
    @unpack T, x, u, dt, states, data, energy, q =  mso_struct
    @unpack data_steps, J = mso_struct

    k = mso_struct.k

    states[:,1] .= x

    kin, ptl, total = ThreeMassSpring.compute_energy(states[:,1], Kc)
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
            # mso_struct.J = mso_struct.J + (E * x - E * data[:, t])' * (E * x - E * data[:, t])

        end

        kin, ptl, total = ThreeMassSpring.compute_energy(x, Kc)
        energy[:,t] = [kin;ptl;total]

    end

    return nothing

end

function grad_descent1(M, params::mso_params_ops, x0::Vector{Float64})

    @unpack T = params

    k_new = 0.0
    params.x .= x0
    params.states .= zeros(6, T+1)
    params.energy .= zeros(3, T+1)

    dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.

    print("Beginning grad descent\n")

    autodiff(Reverse, integrate1, Duplicated(params, dparams))

    if M == 0
        return
    end

    @show dparams.k

    k_new = params.k - (1 / norm(dparams.k)) * dparams.k

    @show k_new

    j = 1
    k_old = copy(params.k)
    k_grad_old = copy(dparams.k)
    params.k = k_new

    J_values = []
    push!(J_values, params.J)
    j_values = []
    push!(j_values, j)

    while norm(k_grad_old) > 500

        params.x .= x0
        params.states .= zeros(6, T+1)
        params.k = k_new
        params.J = 0.0

        dparams = Enzyme.Compiler.make_zero(Core.Typeof(params), IdDict(), params)
        dparams.J = 1.0
        dparams.k = 0.
        dparams.r = 0.
        dparams.dt = 0.
        dparams.Q_inv = 0.

        autodiff(Reverse, integrate1, Duplicated(params, dparams))

        print("Norm of the derivative\n") 
        @show norm(dparams.k)

        print("Current guess for k\n")
        @show params.k

        print("Objective value\n")
        @show params.J

        gamma = 0.0
        num = 0.0
        den = 0.0

        num = sum(dot(params.k - k_old, dparams.k - k_grad_old))
        den = norm(dparams.k - k_grad_old)^2

        gamma = (abs(num) / den)

        k_new = params.k - gamma * dparams.k

        k_old = copy(params.k)
        k_grad_old = copy(dparams.k)
        params.k = k_new

        dparams.k = 0.0

        j += 1

        push!(J_values, params.J)
        push!(j_values, j)

        if j > M
            break
        end

    end

    return params, j_values, J_values

end

function exp_5_param(;optm_steps = 100,k_guess=31.)

    # Parameter choices
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    # q_kf(t) = q_true(t)
    ###########################################
   
    data_steps1 = [k for k in 3000:200:7000]         # steps where data will be assimilated
    # data_steps2 = [k for k in 7200:100:9000]
    # data_steps = [data_steps1;data_steps2]
    data_steps = data_steps1
    # data_steps = [t for t in 1:T]

    rand_forcing = 0.1 .* randn(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing

    params_true = ThreeMassSpring.mso_params(T = T,
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

    params_pred = ThreeMassSpring.mso_params(T = T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    q = q_kf,
    u = u,
    k = k_guess,
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops_true = ThreeMassSpring.build_ops(params_true)
    ops_pred = ThreeMassSpring.build_ops(params_pred)

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
    _ = ThreeMassSpring.create_data(params_pred, ops_pred)

    states_noisy = ThreeMassSpring.create_data(params_true, ops_true)

    diag = 0.0
    Q_inv = diag

    ###################################
    R_inv = ops_pred.R^(-1)
    # R_inv = ops_pred.E
    ###################################

    params_kf = mso_params(T=T,
    x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    u = u,
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

    diag = 0.0
    Q_inv = diag

    ###############################
    R_inv = ops_pred.R^(-1)
    # R_inv = Diagonal(ones(6))
    ##############################

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

    ################# Handwritten gradient descent

    # params_adjoint, j, J = grad_descent1(100, params_adjoint, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # plot(j, J)
    # xlabel!("iteration")
    # ylabel!("Objective value")

    ################### Optim.jl

    # Combined cost and gradient

    # sol = optimize(Optim.only_fg!(cost_gradient_eval),
    #     [k_guess],
    #     Optim.LBFGS(),
    #     Optim.Options(allow_f_increases=true)
    # )

    # Separate cost and gradient 

    sol = optimize(cost_eval, gradient_eval, [k_guess], LBFGS())

    return sol

    # # plot of fixed displacement
    # fixed_pos = plot(params_true.states[2,:],
    # label = L"x_2(t)"
    # )

    # # plot!(params_pred.states[2, :],
    # # label = L"\tilde{x}_2(t, -)"
    # # )
    # # for uncertainty ribbon 
    # to_plot1 = []
    # to_plot2 = []
    # for j = 1:T+1 
    #     push!(to_plot1, sqrt(uncertainty[j][2,2]))
    #     push!(to_plot2, sqrt(uncertainty[j][5,5]))
    # end
    # plot!(params_kf.states[2,:], ribbon = to_plot1, ls=:dash, label = L"\tilde{x}_2(t)")
    
    # plot!(params_adjoint.states[2,:], ls=:dashdot,
    # label = L"\tilde{x}_2(t, +)")
    # vline!(data_steps, 
    # label = "",
    # ls=:dot,
    # lc=:red,
    # lw=0.5
    # )
    # ylabel!("Position")

    # # plot of fixed velocity 
    # fixed_vel = plot(params_true.states[5,:],
    # label = L"x_5(t)")
    # # plot!(params_pred.states[5, :],
    # # label = L"\tilde{x}_5(t, -)"
    # # )
    # plot!(params_kf.states[5,:], ls=:dash, label = L"\tilde{x}_5(t)", ribbon = to_plot2)
    # plot!(params_adjoint.states[5,:], ls=:dashdot,
    # label = L"\tilde{x}_5(t, +)")
    # vline!(data_steps, 
    # label = "",
    # ls=:dot,
    # lc=:red,
    # lw=0.5
    # )
    # ylabel!("Velocity")

    # # plot of energy 
    # energy = plot(params_true.energy[3,:],
    # label=L"\varepsilon(t)")
    # plot!(params_pred.energy[3,:], label=L"\tilde{\varepsilon}(t,-)")
    # plot!(params_kf.energy[3,:], ls=:dash, label=L"\tilde{\varepsilon}(t)")
    # plot!(params_adjoint.energy[3,:], ls=:dashdot, label=L"\tilde{\varepsilon}(t,+)")
    # vline!(data_steps, 
    # label = "",
    # ls=:dot,
    # lc=:red,
    # lw=0.5
    # )
    # ylabel!("Energy")

    # # plot of differences 
    # diffs = plot(abs.(params_true.states[2,:] - params_kf.states[2,:]),
    # label = L"|x_2(t) - \tilde{x}_2(t)|"
    # )
    # plot!(abs.(params_true.states[2, :] - params_adjoint.states[2,:]),
    # label = L"|x_2(t) - \tilde{x}_2(t, +)|"
    # )
    # ylabel!("Position")
    # xlabel!("Timestep")

    # plot(fixed_pos, fixed_vel, energy, diffs,
    # layout = (4,1),
    # fmt = :png,
    # dpi = 300,
    # legend = :outerright)
    

end

### checking the derivative returned by Enzyme

function enzyme_check_param(;k_guess=20.)

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
 
     params_true = ThreeMassSpring.mso_params(T = T,
     x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     u = u,
     k = 30,
     n = 0.0 .* randn(6, T+1),
     q = q_true,
     data_steps = data_steps,
     data = zeros(1,1),
     states = zeros(6, T+1),
     energy = zeros(3, T+1)
     )
 
     params_pred = ThreeMassSpring.mso_params(T = T,
     x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     q = q_kf,
     u = u,
     k = k_guess,
     data_steps = data_steps,
     data = zeros(1,1),
     states = zeros(6, T+1),
     energy = zeros(3, T+1)
     )
    
     ops_true = ThreeMassSpring.build_ops(params_true)
     ops_pred = ThreeMassSpring.build_ops(params_pred)
 
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
     _ = ThreeMassSpring.create_data(params_pred, ops_pred)

     states_noisy = ThreeMassSpring.create_data(params_true, ops_true)

    diag = 0.0
    Q_inv = diag

    ###############################
    # R_inv = ops_pred.R^(-1)
    R_inv = ops_pred.E
    ###############################

    params_adjoint2 = ThreeMassSpring.mso_params_ops(T=T,
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

    dparams_adjoint2 = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint2), IdDict(), params_adjoint2)
    dparams_adjoint2.J = 1.0
    dparams_adjoint2.k = 0.
    dparams_adjoint2.r = 0.
    dparams_adjoint2.dt = 0.
    dparams_adjoint2.Q_inv = 0.

    autodiff(Reverse, integrate1, Duplicated(params_adjoint2, dparams_adjoint2))

    params_adjoint = ThreeMassSpring.mso_params_ops(T=T,
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

     ThreeMassSpring.integrate1(params_adjoint)

    for_checking = 0.0
    for j in data_steps

        for_checking = for_checking + (params_adjoint.states[:,j] - states_noisy[:,j])' * params_adjoint.R_inv * (params_adjoint.states[:,j] - states_noisy[:,j])

    end

    steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    diffs = []
    params_fc = ThreeMassSpring.mso_params_ops(T=T,
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
            end

            temp += params_fc.dt

        end

        @show total_cost

        push!(diffs, (total_cost - for_checking)/s)

    end

    @show diffs
    @show dparams_adjoint2.k


end