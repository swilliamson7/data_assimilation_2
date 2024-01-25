using Optimization, OptimizationOptimJL
using Enzyme

function gradient_eval!(G, k)

    k = k[1]
    # Parameter choices 
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    # q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [j for j in 3000:200:7000]         # steps where data will be assimilated
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
    k = 30.,
    n = 0.0 .* randn(6, T+1), ###################################
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
    k = k_guess[1],
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops_true = ThreeMassSpring.build_ops(params_true)
    ops_pred = ThreeMassSpring.build_ops(params_pred)

    # assuming data of all positions and velocities -> E is the identity operator
    ops_pred.E .= Diagonal(ones(6))
    ops_true.E .= Diagonal(ones(6))

    ops_pred.Q[1,1] = cov(params_true.u[:], corrected=false)
    ops_pred.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))
    ops_true.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops_pred.Gamma[1, 1] = 1.0
    ops_true.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = ThreeMassSpring.create_data(params_pred, ops_pred)

    states_noisy = ThreeMassSpring.create_data(params_true, ops_true)

    Q_inv = 0.0

    ######################
    # R_inv = ops.R^(-1)
    R_inv = ops.E
    ######################

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

    dparams = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.
    @show dparams

    autodiff(Reverse, ThreeMassSpring.integrate1, Duplicated(params_adjoint, dparams))

    @show dparams

    G[1] = dparams.k

end

function cost_and_gradient_eval(F, G, k)

    k_guess = k[1]
    # Parameter choices
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    # q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [j for j in 3000:200:7000]         # steps where data will be assimilated
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
    k = 30.,
    n = 0.0 .* randn(6, T+1), ###################################
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
    k = k_guess[1],
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops_true = ThreeMassSpring.build_ops(params_true)
    ops_pred = ThreeMassSpring.build_ops(params_pred)

    # assuming data of all positions and velocities -> E is the identity operator
    ops_pred.E .= Diagonal(ones(6))
    ops_true.E .= Diagonal(ones(6))

    ops_pred.Q[1,1] = cov(params_true.u[:], corrected=false)
    ops_pred.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))
    ops_true.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops_pred.Gamma[1, 1] = 1.0
    ops_true.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = ThreeMassSpring.create_data(params_pred, ops_pred)

    states_noisy = ThreeMassSpring.create_data(params_true, ops_true)

    Q_inv = 0.0

    ######################
    # R_inv = ops.R^(-1)
    R_inv = ops.E
    ######################

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

    dparams = ThreeMassSpring.Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.

    Enzyme.autodiff(Reverse, ThreeMassSpring.integrate1, Duplicated(params_adjoint, dparams))

    if G !== nothing
        G = dparams.k
    end

    if F !== nothing
        return params_adjoint.J
    end

end

function cost_function_eval(k_guess)

    # Parameter choices 
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    # q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    q_kf(t) = q_true(t)
    ###########################################

    data_steps1 = [j for j in 3000:200:7000]         # steps where data will be assimilated
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
    k = 30.,
    n = 0.0 .* randn(6, T+1), ###################################
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
    k = k_guess[1],
    data_steps = data_steps,
    data = zeros(1,1),
    states = zeros(6, T+1),
    energy = zeros(3, T+1)
    )

    ops_true = ThreeMassSpring.build_ops(params_true)
    ops_pred = ThreeMassSpring.build_ops(params_pred)

    # assuming data of all positions and velocities -> E is the identity operator
    ops_pred.E .= Diagonal(ones(6))
    ops_true.E .= Diagonal(ones(6))

    ops_pred.Q[1,1] = cov(params_true.u[:], corrected=false)
    ops_pred.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))
    ops_true.R .= cov(params_true.n[:], corrected=false) .* Diagonal(ones(6))

    # assuming random forcing on position of mass one
    ops_pred.Gamma[1, 1] = 1.0
    ops_true.Gamma[1, 1] = 1.0

    # pure prediction model
    _ = ThreeMassSpring.create_data(params_pred, ops_pred)

    states_noisy = ThreeMassSpring.create_data(params_true, ops_true)

    Q_inv = 0.0

    ######################
    # R_inv = ops.R^(-1)
    R_inv = ops.E
    ######################

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

    return params_adjoint.J

end

k_guess = [32.]
p = 0.0
optf = OptimizationFunction(cost_function_eval, Optimization.AutoEnzyme())
prob = OptimizationProblem(optf,k_guess,p)
sol = solve(prob, BFGS())

# k_guess = [29.0]
# p = 0.0
# result = Optim.optimize(cost_function_eval, k_guess)
# result = Optim.optimize(Optim.only_fg!(cost_and_gradient_eval), k_guess)

# k_values = 20.0:1.0:40.0
# cost_values = zeros(length(k_values))
# for i = 1:length(k_values)

#     J = cost_function_eval(k_values[i])
#     cost_values[i] = J

# end

# plot(k_values, cost_values)
# xlabel!("k")
# ylabel!("J(k)")