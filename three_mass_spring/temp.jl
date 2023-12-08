using Parameters, LinearAlgebra, Enzyme

@with_kw mutable struct mso_params_ops1{F<:Function}

    T::Int               # Total steps to integrate 
    t::Int = 0           # placeholder for current step
    dt::Float64 = 0.001  # timestep

    x::Vector{Float64} = zeros(6)        # placeholder for state vector 
    u::Matrix{Float64} = zeros(6, T+1)   # random forcing (if any), otherwise just leave zero 
    n::Matrix{Float64} = zeros(6, T+1)   # placeholder for noise to add to the data 

    k::Float64 = 30          # spring constant
    r::Float64 = 0.5         # Rayleigh friction coefficient

    q::F                     # forcing function 

    J::Float64 = 0.0         # cost function evaluation 

    data_steps::Vector{Int64}  # the timesteps where data points exist 
    data::Matrix{Float64}

    states::Matrix{Float64}    # placeholder for computed states 
    energy::Matrix{Float64}    # placeholder for computed energy

    A::Matrix{Float64} = zeros(6,6)                            # Time-step (x(t+1) = A x(t))
    B::Matrix{Float64} = diagm([1., 0., 0., 0., 0., 0.])       # Distributes known forcing  
    Gamma::Matrix{Float64} = zeros(6,6)                        # Distrubutes unknown (random) forcing

    E::Matrix{Float64} = zeros(6,6)           # Acts on data vector, generally the identity (e.g. full info on all positions/velocities)

    Q::Matrix{Float64} = zeros(6,6)           # Covariance matrix for unknown (random) forcing
    Q_inv::Float64
    R::Matrix{Float64} = zeros(6,6)           # Covariance matrix for noise in data
    R_inv::Matrix{Float64}                    # Inverse of operator R

    K::Matrix{Float64} = zeros(6,6)           # Placeholder for Kalman gain matrix 
    Kc::Matrix{Float64} = zeros(6,6)     

end

function run(;k=20)

    # Parameter choices
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
    q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
    data_steps1 = [k for k in 3000:200:7000]         # steps where data will be assimilated
    data_steps = data_steps1
    dt = 1

    rand_forcing = 0.1 .* ones(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing
    n = 0.05 .* ones(6, T+1)

    Q = zeros(6,6)
    K = zeros(6,6)
    Gamma = zeros(6,6)
    for_building = [1. 0. 0.
                    0. 1. 0.
                    0. 0. 1.]
    Rc = -r .* for_building
    Kc = [-2.0*k k 0.0; k -3.0*k k; 0.0 k -2.0*k]
    Ac = [zeros(3,3) for_building
        Kc Rc
    ]

    A = [1. 0. 0. 0. 0. 0.
        0. 1. 0. 0. 0. 0.
        0. 0. 1. 0. 0. 0.
        0. 0. 0. 1. 0. 0.
        0. 0. 0. 0. 1. 0.
        0. 0. 0. 0. 0. 1.]  + dt .* Ac

    Q[1,1] = 0.2
    R = [1. 0. 0. 0. 0. 0.
        0. 1. 0. 0. 0. 0.
        0. 0. 1. 0. 0. 0.
        0. 0. 0. 1. 0. 0.
        0. 0. 0. 0. 1. 0.
        0. 0. 0. 0. 0. 1.]

    # assuming random forcing on position of mass one 
    Gamma[1, 1] = 1.0 

    # create data from the true setup
    states_noisy = zeros(6, T+1)

    B = [1. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.]

    Gamma = B 

    E = [1. 0. 0. 0. 0. 0.
        0. 1. 0. 0. 0. 0.
        0. 0. 1. 0. 0. 0.
        0. 0. 0. 1. 0. 0.
        0. 0. 0. 0. 1. 0.
        0. 0. 0. 0. 0. 1.]

    params_adjoint2 = mso_params_ops1(T=T,
        t = 0,
        x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        u = u,
        n = 0.001 .* randn(6, T+1),
        q = q_kf,
        J = 0.0,
        k = k,
        data_steps = data_steps,
        data = states_noisy,
        states = zeros(6, T+1),
        energy = zeros(3, T+1),
        A = A,
        B = B,
        Gamma = Gamma,
        E = E,
        Q = 0.0 .* Q,
        Q_inv = 1.,
        R = R,
        R_inv = diagm(ones(6)),
        K = K,
        Kc = Kc
    )

    dparams_adjoint2 = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint2), IdDict(), params_adjoint2)
    dparams_adjoint2.J = 1.0

    @show dparams_adjoint2.k

end

function enzyme_check_param(;k_guess=20.)

    # Parameter choices
    T = 10000             # Total number of steps to integrate
    r = 0.5               # spring coefficient
    q_true(t) = 0.1 * cos(2 * pi * t / (2.5 / r))      # known forcing function
   #  q_kf(t) = 0.5 * q_true(t)                          # forcing seen by KF and adjoint
   q_kf(t) = q_true(t)
   data_steps1 = [k for k in 3000:200:7000]         # steps where data will be assimilated
    data_steps = data_steps1

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

    rand_forcing = 0.1 .* ones(T+1)
    u = zeros(6, T+1)
    u[1, :] .= rand_forcing
    n = 0.05 .* ones(6, T+1)

    Q = zeros(6,6)
    K = zeros(6,6)
    Gamma = zeros(6,6)
    for_building = [1. 0. 0.
                    0. 1. 0.
                    0. 0. 1.]
    Rc = -r .* for_building
    Kc = [-2.0*k k 0.0; k -3.0*k k; 0.0 k -2.0*k]
    Ac = [zeros(3,3) for_building
        Kc Rc
    ]

    A = [1. 0. 0. 0. 0. 0.
        0. 1. 0. 0. 0. 0.
        0. 0. 1. 0. 0. 0.
        0. 0. 0. 1. 0. 0.
        0. 0. 0. 0. 1. 0.
        0. 0. 0. 0. 0. 1.]  + dt .* Ac

    Q[1,1] = 0.2
    R = 0.1 .* [1. 0. 0. 0. 0. 0.
                0. 1. 0. 0. 0. 0.
                0. 0. 1. 0. 0. 0.
                0. 0. 0. 1. 0. 0.
                0. 0. 0. 0. 1. 0.
                0. 0. 0. 0. 0. 1.] 

    # assuming random forcing on position of mass one 
    Gamma[1, 1] = 1.0 

    # create data from the true setup
    states_noisy = zeros(6, T+1)

    B = [1. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0.]

    Gamma = B 

    E = [1. 0. 0. 0. 0. 0.
        0. 1. 0. 0. 0. 0.
        0. 0. 1. 0. 0. 0.
        0. 0. 0. 1. 0. 0.
        0. 0. 0. 0. 1. 0.
        0. 0. 0. 0. 0. 1.]


   # just Enzyme
   params_adjoint2 = mso_params_ops(T=T,
       t = 0,
       x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       u = u,
       q = q_kf,
       J = 0.0,
       k = k_guess,
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

   dparams_adjoint2 = Enzyme.Compiler.make_zero(Core.Typeof(params_adjoint2), IdDict(), params_adjoint2)
   dparams_adjoint2.J = 1.0
   dparams_adjoint2.k = 0.
   dparams_adjoint2.r = 0.
   dparams_adjoint2.dt = 0.

   autodiff(Reverse, integrate1, Duplicated(params_adjoint2, dparams_adjoint2))

   # no Enzyme, manual cost function evaluation
   params_adjoint = mso_params_ops(T=T,
       t = 0,
       x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       u = u,
       q = q_kf,
       J = 0.0,
       k = k_guess,
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

   integrate1(params_adjoint)

   for_checking = 0.0
   for j in data_steps

       for_checking = for_checking + (params_adjoint.states[:,j] - states_noisy[:,j])' * R_inv * (params_adjoint.states[:,j] - states_noisy[:,j])

   end

   steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
   diffs = []
   params_fc = mso_params_ops(T=T,
       t = 0,
       x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       u = u,
       q = q_kf,
       J = 0.0,
       k = k_guess,
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
               total_cost = total_cost + (params_fc.x - states_noisy[:,j])' * R_inv * (params_fc.x - states_noisy[:,j]) 
           end

           temp += params_fc.dt

       end

       @show total_cost

       push!(diffs, (total_cost - for_checking)/s)

   end

   @show diffs
   @show dparams_adjoint2.k


end