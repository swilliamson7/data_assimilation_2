function setup_model(;k_guess = [31.], T = 10000)

    # Parameter choices
    T = T             # Total number of steps to integrate
    r = 0.5               # spring coefficient

    ###########################################
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
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
    k = k_guess[1],
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

    ###################################
    R_inv = ops_pred.R^(-1)
    # R_inv = ops_pred.E
    ###################################

    params_kf = mso_params(T=T,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        k = k_guess[1],
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
        u = u,
        n = 0.0 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k_guess[1],
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

    return params_adjoint, params_pred, params_true, params_kf, uncertainty

end

function cost_eval(k_guess, params_adjoint)

    T = params_adjoint.T
    params_adjoint.k = k_guess[1]
    params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.states .= zeros(6, T+1)
    params_adjoint.J = 0.0
    params_adjoint.energy .= zeros(3, T+1)

    integrate1(params_adjoint)

    return params_adjoint.J

end

function gradient_eval(G, k_guess, params_adjoint)

    T = params_adjoint.T

    params_adjoint.k = k_guess[1]
    params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    params_adjoint.states .= zeros(6, T+1)
    params_adjoint.J = 0.0
    params_adjoint.energy .= zeros(3, T+1)

    dparams = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
    dparams.J = 1.0
    dparams.k = 0.
    dparams.r = 0.
    dparams.dt = 0.
    dparams.Q_inv = 0.

    autodiff(Reverse, integrate1, Duplicated(params_adjoint, dparams))

    G[1] = dparams.k

end

function FG(F, G, k_guess, params_adjoint)

    G === nothing || gradient_eval(G, k_guess, params_adjoint)
    F === nothing || return cost_eval(k_guess, params_adjoint)

end

# function cost_gradient(F, G, k_guess, params_adjoint)

#     T = params_adjoint.T

#     params_adjoint.k = k_guess[1]
#     params_adjoint.x .= [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     params_adjoint.states .= zeros(6, T+1)
#     params_adjoint.J = 0.0
#     params_adjoint.energy .= zeros(3, T+1)

#     dparams = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint), IdDict(), params_adjoint)
#     dparams.J = 1.0
#     dparams.k = 0.
#     dparams.r = 0.
#     dparams.dt = 0.
#     dparams.Q_inv = 0.

#     autodiff(Reverse, integrate1, Duplicated(params_adjoint, dparams))

#     F[1] = params_adjoint.J
#     G[1] = dparams.k

#     return nothing

# end

# without gradient, seems to work
# params_adjoint, params_pred, params_true = ThreeMassSpring.setup_model()
# sol = optimize(k -> ThreeMassSpring.cost_eval(k, params_adjoint), [31.], Optim.LBFGS())

# # trying to get a method with the gradient
# sol = optimize(k -> Optim.only_fg(ThreeMassSpring.cost_gradient_eval(F, G, k, params_adjoint)), [31.], Optim.LBFGS())
# sol = optimize(k -> ThreeMassSpring.cost_eval(k, params_adjoint), k -> ThreeMassSpring.gradient_eval(G, k, params_adjoint), [31.], Optim.LBFGS())
# sol = optimize(Optim.only_fg((F, G, k) -> ThreeMassSpring.cost_gradient_eval(F, G, k, params_adjoint)), [31.], Optim.LBFGS())
# obj(F, k) = ThreeMassSpring.cost_gradient_eval(F, G, k, params_adjoint)
# optimize(obj, [31.], Optim.LBFGS())

# WORKING

params_adjoint, params_pred, params_true = setup_model();
fg!_closure(F, G, k) = FG(F, G, k, params_adjoint)
obj_fg = Optim.only_fg!(fg!_closure)
result = Optim.optimize(obj_fg, [27.], Optim.LBFGS(), Optim.Options(show_trace=true))