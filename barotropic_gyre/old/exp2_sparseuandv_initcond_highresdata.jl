mutable struct exp2_hradj_model{T, S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    S::ShallowWaters.ModelSetup{T,T}        # model struct for adjoint
    data::Array{T, 2}                       # data to be assimilated
    data_steps::StepRange{Int, Int}         # when data is assimilated
    data_spots::Array{Int, 1}               # where data is located, grid coordinates
    J::Float64                              # objective value
    j::Int                                  # for keeping track of location in data
    i::Int                                  # timestep iterator
    t::Int64                                # model time
end

mutable struct exp2_hrekf_model{T}
    S::ShallowWaters.ModelSetup{T,T}        # model struct for adjoint
    N::Int                                  # number of ensemble members
    data::Array{T, 2}                       # data to be assimilated
    sigma_initcond::T
    sigma_data::T
    data_steps::StepRange{Int, Int}         # when data is assimilated
    data_spots::Array{Int, 1}               # where data is located, grid coordinates
    j::Int                                  # for keeping track of location in data
end


function exp2hr_coarse_grain(u_hr, v_hr, η_hr, nx_hr, ny_hr, S_lr)

    σ = 30

    u_lr = S_lr.Prog.u
    v_lr = S_lr.Prog.v
    η_lr = S_lr.Prog.η

    mu, nu = size(u_lr)
    mu_hr, nu_hr = size(u_hr)
    mv, nv = size(v_lr)
    mv_hr, nv_hr = size(v_hr)
    meta, neta = size(η_lr)
    meta_hr, neta_hr = size(η_hr)

    ū_hr = zeros(mu, nu)
    v̄_hr = zeros(mv, nv)
    η̅_hr = zeros(meta, neta)

    # needs to be adjusted for any grid that isn't square
    x_lr = 0:S_lr.grid.Δ:S_lr.parameters.Lx
    y_lr = 0:S_lr.grid.Δ:(S_lr.parameter.Lx/S_lr.parameters.L_ratio)

    Δ_hr = S_lr.parameters.Lx / nx_hr

    x_hr = 0:Δ_hr:S_lr.parameters.Lx
    y_hr = 0:Δ_hr:(S_lr.parameters.Lx/S_lr.parameters.L_ratio)

    coeff = 1 / (2 * π * σ^2)

    for j ∈ nu, k ∈ mu

        for j2 ∈ nu_hr
            for k2 ∈ mu_hr

                ū_hr[k,j] = ū_hr[k,j] + u_hr[k2,j2] * exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/(2 * σ^2))

            end
        end

        ū_hr = coeff * ū_hr

    end

    for j ∈ 1:nv, k ∈ 1:mv

        for j2 ∈ 1:nv_hr
            for k2 ∈ 1:mv_hr

                v̄_hr[k,j] = v̄_hr[k,j] + v_hr[k2,j2] * exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/(2 * σ^2))

            end
        end

        v̄_hr = coeff * v̄_hr

    end

    for j ∈ 1:neta, k ∈ 1:meta

        for j2 ∈ 1:neta_hr
            for k2 ∈ 1:meta_hr

                η̅_hr[k,j] = η̅_hr[k,j] + η_hr[k2,j2] * exp(-((x_hr[j2] - x_lr[j])^2 + (y_hr[k2] - y_lr[k])^2)/(2 * σ^2))

            end
        end

        η̅_hr = coeff * η̅_hr

    end

    return ū_hr, v̄_hr, η̅_hr

end

function exp2hr_cpintegrate(chkp, scheme)::Float64

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(chkp.S.Diag.VolumeFluxes.h, chkp.S.Prog.η, chkp.S.forcing.H)
    ShallowWaters.Ix!(chkp.S.Diag.VolumeFluxes.h_u, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(chkp.S.Diag.VolumeFluxes.h_v, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(chkp.S.Diag.Vorticity.h_q, chkp.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Prog.u)
    vrhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Prog.v)
    ηrhs = convert(chkp.S.Diag.PrognosticVarsRHS.η, chkp.S.Prog.η)

    ShallowWaters.advection_coriolis!(urhs, vrhs, ηrhs, chkp.S.Diag, chkp.S)
    ShallowWaters.PVadvection!(chkp.S.Diag, chkp.S)

    # propagate initial conditions
    copyto!(chkp.S.Diag.RungeKutta.u0, chkp.S.Prog.u)
    copyto!(chkp.S.Diag.RungeKutta.v0, chkp.S.Prog.v)
    copyto!(chkp.S.Diag.RungeKutta.η0, chkp.S.Prog.η)

    # store initial conditions of sst for relaxation
    copyto!(chkp.S.Diag.SemiLagrange.sst_ref, chkp.S.Prog.sst)

    # run integration loop with checkpointing
    chkp.j = 1
    @ad_checkpoint scheme for chkp.i = 1:chkp.S.grid.nt

        t = chkp.S.t
        i = chkp.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(chkp.S.Prog.u, chkp.S.Prog.v, chkp.S.Prog.η, chkp.S)
        copyto!(chkp.S.Diag.RungeKutta.u1, chkp.S.Prog.u)
        copyto!(chkp.S.Diag.RungeKutta.v1, chkp.S.Prog.v)
        copyto!(chkp.S.Diag.RungeKutta.η1, chkp.S.Prog.η)

        if chkp.S.parameters.compensated
            fill!(chkp.S.Diag.Tendencies.du_sum, zero(chkp.S.parameters.Tprog))
            fill!(chkp.S.Diag.Tendencies.dv_sum, zero(chkp.S.parameters.Tprog))
            fill!(chkp.S.Diag.Tendencies.dη_sum, zero(chkp.S.parameters.Tprog))
        end

        for rki = 1:chkp.S.parameters.RKo
            if rki > 1
                ShallowWaters.ghost_points!(
                    chkp.S.Diag.RungeKutta.u1,
                    chkp.S.Diag.RungeKutta.v1,
                    chkp.S.Diag.RungeKutta.η1,
                    chkp.S
                )
            end

            # type conversion for mixed precision
            u1rhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Diag.RungeKutta.u1)
            v1rhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Diag.RungeKutta.v1)
            η1rhs = convert(chkp.S.Diag.PrognosticVarsRHS.η, chkp.S.Diag.RungeKutta.η1)

            ShallowWaters.rhs!(u1rhs, v1rhs, η1rhs, chkp.S.Diag, chkp.S, t)          # momentum only
            ShallowWaters.continuity!(u1rhs, v1rhs, η1rhs, chkp.S.Diag, chkp.S, t)   # continuity equation

            if rki < chkp.S.parameters.RKo
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.u1,
                    chkp.S.Prog.u,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.du
                )
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.v1,
                    chkp.S.Prog.v,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.dv
                )
                ShallowWaters.caxb!(
                    chkp.S.Diag.RungeKutta.η1,
                    chkp.S.Prog.η,
                    chkp.S.constants.RKbΔt[rki],
                    chkp.S.Diag.Tendencies.dη
                )
            end

            if chkp.S.parameters.compensated
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.du_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.du)
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.dv_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.dv)
                ShallowWaters.axb!(chkp.S.Diag.Tendencies.dη_sum, chkp.S.constants.RKaΔt[rki], chkp.S.Diag.Tendencies.dη)
            else
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.u0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.du
                )
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.v0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.dv
                )
                ShallowWaters.axb!(
                    chkp.S.Diag.RungeKutta.η0,
                    chkp.S.constants.RKaΔt[rki],
                    chkp.S.Diag.Tendencies.dη
                )
            end
        end

        if chkp.S.parameters.compensated
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.du_sum, -1, chkp.S.Diag.Tendencies.du_comp)
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.dv_sum, -1, chkp.S.Diag.Tendencies.dv_comp)
            ShallowWaters.axb!(chkp.S.Diag.Tendencies.dη_sum, -1, chkp.S.Diag.Tendencies.dη_comp)

            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.u0, 1, chkp.S.Diag.Tendencies.du_sum)
            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.v0, 1, chkp.S.Diag.Tendencies.dv_sum)
            ShallowWaters.axb!(chkp.S.Diag.RungeKutta.η0, 1, chkp.S.Diag.Tendencies.dη_sum)

            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.du_comp,
                chkp.S.Diag.RungeKutta.u0,
                chkp.S.Prog.u,
                chkp.S.Diag.Tendencies.du_sum
            )
            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.dv_comp,
                chkp.S.Diag.RungeKutta.v0,
                chkp.S.Prog.v,
                chkp.S.Diag.Tendencies.dv_sum
            )
            ShallowWaters.dambmc!(
                chkp.S.Diag.Tendencies.dη_comp,
                chkp.S.Diag.RungeKutta.η0,
                chkp.S.Prog.η,
                chkp.S.Diag.Tendencies.dη_sum
            )
        end

        ShallowWaters.ghost_points!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S.Diag.RungeKutta.η0,
            chkp.S
        )

        u0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Diag.RungeKutta.u0)
        v0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Diag.RungeKutta.v0)
        η0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.η, chkp.S.Diag.RungeKutta.η0)

        if chkp.S.parameters.dynamics == "nonlinear" && chkp.S.grid.nstep_advcor > 0 && (i % chkp.S.grid.nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
            ShallowWaters.advection_coriolis!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
        end

        if (chkp.S.parameters.i % chkp.S.grid.nstep_diff) == 0
        ShallowWaters.bottom_drag!(u0rhs, v0rhs, η0rhs, chkp.S.Diag, chkp.S)
        ShallowWaters.diffusion!(u0rhs, v0rhs, chkp.S.Diag, chkp.S)
        ShallowWaters.add_drag_diff_tendencies!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S.Diag,
            chkp.S
        )
        ShallowWaters.ghost_points_uv!(
            chkp.S.Diag.RungeKutta.u0,
            chkp.S.Diag.RungeKutta.v0,
            chkp.S
        )
    end

    t += chkp.S.grid.dtint

    u0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.u, chkp.S.Diag.RungeKutta.u0)
    v0rhs = convert(chkp.S.Diag.PrognosticVarsRHS.v, chkp.S.Diag.RungeKutta.v0)
    ShallowWaters.tracer!(i, u0rhs, v0rhs, chkp.S.Prog, chkp.S.Diag, chkp.S)

    if i in chkp.data_steps
        temp = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(
            chkp.S.Prog.u,
            chkp.S.Prog.v,
            chkp.S.Prog.η,
            chkp.S.Prog.sst,
            chkp.S
        )...)

        tempuv = [vec(temp.u); vec(temp.v)][chkp.data_spots]

        chkp.J += sum((tempuv - chkp.data[:, chkp.j]).^2)

        chkp.j += 1
    end

    copyto!(chkp.S.Prog.u, chkp.S.Diag.RungeKutta.u0)
    copyto!(chkp.S.Prog.v, chkp.S.Diag.RungeKutta.v0)
    copyto!(chkp.S.Prog.η, chkp.S.Diag.RungeKutta.η0)
    end

    return chkp.J

end

function exp2hr_cost_eval(param_guess, data, data_spots, data_steps, Ndays)

    # Type precision
    T = Float64

    P = ShallowWaters.Parameter(T=T;
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    S = ShallowWaters.model_setup(P)
    current = 1
    for m in (S.Prog.u, S.Prog.v, S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    data_spots = Int.(data_spots)

    chkp = exp2hr_model{T}(S,
        data,
        data_spots,
        data_steps,
        0.0,
        1,
        1,
        0.0
    )

    exp2hr_integrate(chkp)

    return chkp.J

end

function exp2hr_gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

    # Type precision
    T = Float64

    P = ShallowWaters.Parameter(T=T;
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    S = ShallowWaters.model_setup(P)

    current = 1
    for m in (S.Prog.u, S.Prog.v, S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{exp2hr_model{T}}(S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )
    data_spots = Int.(data_spots)

    chkp = exp2hr_model{T}(S,
        data,
        data_spots,
        data_steps,
        0.0,
        1,
        1,
        0.0
    )
    dchkp = Enzyme.make_zero(chkp)

    chkp_prim = deepcopy(chkp)
    J = exp2hr_cpintegrate(chkp_prim, revolve)
    println("Cost without AD: $J")

    @time J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        exp2hr_cpintegrate,
        Active,
        Duplicated(chkp, dchkp),
        Const(revolve)
    )[2]
    println("Cost with AD: $J")

    # Get gradient
    @unpack u, v, η = dchkp.S.Prog
    G .= [vec(u); vec(v); vec(η)]

    return nothing

end

function exp2hr_FG(F, G, param_guess, data, data_spots, data_steps, Ndays)

    G === nothing || exp2hr_gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)
    F === nothing || return exp2hr_cost_eval(param_guess, data, data_spots, data_steps, Ndays)

end

function run_exp2hr_initialcond_uvdata()

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    N = 10
    sigma_data = 0.5        # this and the init cond are what the result from optimization used, be careful in adjusting
    sigma_initcond = .01
    data_steps = 1:225:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]
    Ndays = 30

    P_pred = ShallowWaters.Parameter(T=Float32; output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=30,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    S_true = ShallowWaters.model_setup(P_pred)
    Ndays = S_true.parameters.Ndays

    _, true_states = exp2_generate_data(S_true, data_spots, sigma_data)

    S_temp = ShallowWaters.model_setup(P_pred)
    S_temp.parameters.Ndays = 3
    ShallowWaters.time_integration(S_temp)

    S_pred = ShallowWaters.model_setup(output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="rest"
    )

    uic,vic,etaic = ShallowWaters.add_halo(Prog_pred.u,Prog_pred.v,Prog_pred.η,Prog_pred.sst,S_pred)
    S_pred.Prog.u = uic
    S_pred.Prog.v = vic
    S_pred.Prog.η = etaic

    # integrating just the prediction model
    _, pred_states = exp2_generate_data(deepcopy(S_pred), data_spots, sigma_data)

    param_guess = [vec(uic); vec(vic); vec(etaic)]

    _, ekf_avgu, ekf_avgv = run_ensemble_kf(N,
        data,
        copy(param_guess),
        data_spots,
        sigma_initcond,
        sigma_data;
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    dS = Enzyme.Compiler.make_zero(S_pred)
    G = zeros(length(dS.Prog.u) + length(dS.Prog.v) + length(dS.Prog.η))

    # Ndays = copy(P_pred.Ndays)
    # data_steps = copy(P_pred.data_steps)
    # exp2_gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

    fg!_closure(F, G, ic) = exp2_FG(F, G, ic, data, data_spots, data_steps, Ndays)
    obj_fg = Optim.only_fg!(fg!_closure)
    result = Optim.optimize(obj_fg, copy(param_guess), Optim.LBFGS(), Optim.Options(show_trace=true, iterations=50))

    S_adj = ShallowWaters.model_setup(output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        tracer_advection=false,
        tracer_relaxation=false,
        bottom_drag="quadratic",
        seasonal_wind_x=false,
        data_steps=data_steps,
        topography="flat",
        bc="nonperiodic",
        α=2,
        nx=128,
        Ndays=Ndays
    )
    current = 1
    for m in (S_adj.Prog.u, S_adj.Prog.v, S_adj.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    # uad = JLD2.load("exp2_minimizer_initcond_adjoint_042525.jld2")["u"]
    # vad = JLD2.load("exp2_minimizer_initcond_adjoint_042525.jld2")["v"]
    # etaad = JLD2.load("exp2_minimizer_initcond_adjoint_042525.jld2")["eta"]
    # S_adj.Prog.u = uad
    # S_adj.Prog.v = vad
    # S_adj.Prog.η = etaad

    _, states_adj = exp2_generate_data(S_adj, data_spots, sigma_data)

    return result, pred_states, ekf_avgu, ekf_avgv, data, true_states, S_adj, states_adj
end

function exp2_hr_plots()

    # S_kf_all, Progkf_all, G, dS, data, true_states, result, S_adj, states_adj

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    N = 10
    sigma_data = 0.5        # this and the init cond are what the result from optimization used, be careful in adjusting
    sigma_initcond = .01
    data_steps = 225:225:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]
    Ndays = 30

    fig1 = Figure(size=(1000, 500));
    ax1, hm1 = heatmap(fig1[1,1], true_states[end].u,
        colormap=:balance,
        colorrange=(-maximum(true_states[end].u),
        maximum(true_states[end].u)),
        axis=(xlabel=L"x", ylabel=L"y", title=L"u(t = 30 \; \text{days}, x, y)"),
    );
    scatter!(ax1, vec(Xu), vec(Yu), color=:green);
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], true_states[end].v,
        colormap=:balance,
        colorrange=(-maximum(true_states[end].v),
        maximum(true_states[end].v)),
        axis=(xlabel=L"x", ylabel=L"y", title=L"v(t = 30 \; \text{days}, x, y)"),
    );
    scatter!(ax2, vec(Xu), vec(Yu), color=:green);
    Colorbar(fig1[1,4], hm2);


    fig1 = Figure(size=(800,700));
    t = 1
    ax1, hm1 = heatmap(fig1[1,1], states_adj[t].v,
    colormap=:balance,
    colorrange=(-maximum(states_adj[t].v),
    maximum(states_adj[t].v)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y, +)")
    );
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], abs.(true_states[t].v .- states_adj[t].v),
    colormap=:amp,
    # colorrange=(0,
    # 1.5),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|v(x,y) - \tilde{v}(x, y, +)|")
    )
    Colorbar(fig1[1,4], hm2)

    ax3, hm3 = heatmap(fig1[2, 1], ekf_avgv[t],
    colormap=:balance,
    colorrange=(-maximum(ekf_avgv[t]),
    maximum(ekf_avgv[t])),
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y)")
    );
    Colorbar(fig1[2,2], hm3)
    ax4, hm4 = heatmap(fig1[2,3], abs.(true_states[t].v .- ekf_avgv[t]),
    colormap=:amp,
    # colorrange=(0,
    # 1.5),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|v(x,y) - \tilde{v}(x, y)|")
    )
    Colorbar(fig1[2,4], hm4)

    # energy plots

    fig1 = Figure(size=(600, 500));
    ax1, hm1 = heatmap(fig1[1,1], (true_states[end].u[:, 1:end-1].^2 .+ true_states[end].v[1:end-1, :].^2),
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\mathcal{E}"),
    colorrange=(0,
    maximum(true_states[end].u[:, 1:end-1].^2 .+ true_states[end].v[1:end-1, :].^2))
    );
    Colorbar(fig1[1,2], hm1)

    t = 673

    fig1 = Figure(size=(800, 700));
    ax1, hm1 = heatmap(fig1[1,1], states_adj[t].u[:, 1:end-1].^2 .+ states_adj[t].v[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}(+)"),
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    );
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], abs.((true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2) .- (states_adj[t].u[:, 1:end-1].^2 .+ states_adj[t].v[1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}(+)|")
    )
    Colorbar(fig1[1,4], hm2)

    ax3, hm3 = heatmap(fig1[2, 1], ekf_avgu[t][:, 1:end-1].^2 .+ ekf_avgv[t][1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}"),
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    );
    Colorbar(fig1[2,2], hm3)
    ax4, hm4 = heatmap(fig1[2,3], abs.((true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2) .- (ekf_avgu[t][:, 1:end-1].^2 .+ ekf_avgv[t][1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}|")
    )
    Colorbar(fig1[2,4], hm4)

    # frequency wavenumber plots

    fig2 = Figure(size=(800, 500));

    ax1 = Axis(fig2[1,1])

    up_adj = zeros(65, 673)
    vp_adj = zeros(65, 673)

    up_ekf = zeros(65, 673)
    vp_ekf = zeros(65, 673)

    up_true = zeros(65, 673)
    vp_true = zeros(65, 673)

    up_pred = zeros(65, 673)
    vp_pred = zeros(65, 673)

    for t = 1:673
        up_adj[:,t] = power(periodogram(states_adj[t].u; radialavg=true))
        vp_adj[:,t] = power(periodogram(states_adj[t].v; radialavg=true))

        up_ekf[:,t] = power(periodogram(ekf_avgu[t]; radialavg=true))
        vp_ekf[:,t] = power(periodogram(ekf_avgv[t]; radialavg=true))

        up_true[:,t] = power(periodogram(true_states[t].u; radialavg=true))
        vp_true[:,t] = power(periodogram(true_states[t].v; radialavg=true))

        up_pred[:,t] = power(periodogram(pred_states[t].u; radialavg=true))
        vp_pred[:,t] = power(periodogram(pred_states[t].v; radialavg=true))
    end

    fftu_adj = fft(up_adj,[2])
    fftv_adj = fft(vp_adj,[2])
    adj_wl = 1 ./ freq(periodogram(states_adj[3].u; radialavg=true));
    adju_freq = LinRange(0, 672, 673)
    adju_freq = adju_freq ./ 673
    adju_freq = 1 ./ adju_freq 
    adju_freq[1] =  1000
    adj_wl[1] = 1000

    fftu_ekf = fft(up_ekf,[2])
    fftv_ekf = fft(vp_ekf,[2])
    ekf_wl = 1 ./ freq(periodogram(ekf_avgu[3]; radialavg=true));
    ekfu_freq = LinRange(0, 672, 673)
    ekfu_freq = ekfu_freq ./ 673
    ekfu_freq = 1 ./ ekfu_freq 
    ekfu_freq[1] =  1000
    ekf_wl[1] = 1000

    fftu_true = fft(up_ekf,[2])
    fftv_true = fft(vp_ekf,[2])
    true_wl = 1 ./ freq(periodogram(true_states[3].u; radialavg=true));
    trueu_freq = LinRange(0, 672, 673)
    trueu_freq = trueu_freq ./ 673
    trueu_freq = 1 ./ trueu_freq 
    trueu_freq[1] =  1000
    true_wl[1] = 1000

    fftu_pred = fft(up_ekf,[2])
    fftv_pred = fft(vp_ekf,[2])
    pred_wl = 1 ./ freq(periodogram(true_states[3].u; radialavg=true));
    predu_freq = LinRange(0, 672, 673)
    predu_freq = trueu_freq ./ 673
    predu_freq = 1 ./ trueu_freq 
    predu_freq[1] =  1000
    pred_wl[1] = 1000

    heatmap(adj_wl, adju_freq, abs.(fftu_adj) + abs.(fftv_adj); colorscale=log10, axis=(xscale=log10, yscale=log10))

    # one dimensional figures
    fig2 = Figure(size=(800, 500));
    t = 673
    lines(fig2[1,1], adj_wl[2:end], up_adj[2:end,t] + vp_adj[2:end,t], label="Adjoint", axis=(xscale=log10,yscale=log10,xlabel="Wavelength (km)", ylabel="KE(k)", xreversed=true, xticks=[100, 30, 10, 2]))
    lines!(fig2[1,1], ekf_wl[2:end], up_ekf[2:end,t] + vp_ekf[2:end,t], label="EKF")
    lines!(fig2[1,1], true_wl[2:end], up_true[2:end,t] + vp_true[2:end,t], label="Truth")
    lines!(fig2[1,1], pred_wl[2:end], up_pred[2:end,t] + vp_pred[2:end,t], linestyle=:dash, label="Prediction")
    axislegend()

    # time-averaged wavenumber spectrum
    fig2 = Figure(size=(800, 500));
    lines(fig2[1,1], adj_wl[2:end], vec(sum(up_adj[2:end,:] + vp_adj[2:end,:], dims=2)) ./ 64,
        label="Adjoint",
        axis=(xscale=log10,yscale=log10,xlabel="Wavelength (km)", ylabel="KE(k)", xreversed=true, xticks=[100, 30, 10, 2])
    )
    lines!(fig2[1,1], ekf_wl[2:end], vec(sum(up_ekf[2:end,:] + vp_ekf[2:end,:], dims=2))./ 64, label="EKF")
    lines!(fig2[1,1], true_wl[2:end], vec(sum(up_true[2:end,:] + vp_true[2:end,:], dims=2)) ./64 , label="Truth")
    lines!(fig2[1,1], pred_wl[2:end], up_pred[2:end,t] + vp_pred[2:end,t], linestyle=:dash, label="Prediction")
    axislegend()

    fig2 = Figure(size=(800, 500));

    t = 1
    lines(fig2[1,1], adj_wl[2:end], abs.(fft(up_adj[2:end,t])) + abs.(fft(vp_adj[2:end,t])), axis=(xscale=log10,yscale=log10))
    lines!(fig2[1,1], ekf_wl[2:end], abs.(fft(up_ekf[2:end,t])) + abs.(fft(vp_ekf[2:end,t])))
    lines!(fig2[1,1], true_wl[2:end], abs.(fft(up_true[2:end,t])) + abs.(fft(vp_true[2:end,t])))

    # two dimensional freq power plot

    adjmatu = zeros(65, 673)
    adjmatv = zeros(65, 673)

    for j = 1:673
        adjmatu[:, j] = adjfreqpoweru[j].power
        adjmatv[:, j] = adjfreqpowerv[j].power
    end

    # for presentation

    scale_inv = S_adj.constants.scale_inv
    halo = S_adj.grid.halo
    du = scale_inv*tempu[halo+1:end-halo,halo+1:end-halo]
    fig = Figure(size=(700,600), fontsize = 26);
    ax1, hm1 = heatmap(fig[1,1], du, colormap=:balance,colorrange=(-maximum(du), maximum(du)),axis=(xlabel=L"x", ylabel=L"y", title=L"\partial J / \partial u(t_0, x, y)"))
    Colorbar(fig[1,2], hm1);
    fig

end

########################################################################
# Derivative check

function exp2_finite_difference_only(data_spots,x_coord,y_coord;kwargs...)

    T = Float32

    data_steps = 220:220:6733

    P1 = ShallowWaters.Parameter(T=Float32;kwargs...)

    S_outer = ShallowWaters.model_setup(P1)

    data, _ = exp2_generate_data(S_outer, data_spots, 0.01)

    snaps = Int(floor(sqrt(S_outer.grid.nt)))
    revolve = Revolve{exp2_Chkp{T, T}}(S_outer.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )
    data_spots = Int.(data_spots)

    chkp = exp2_Chkp{T, T}(S_outer,
        data,
        data_spots,
        data_steps,
        0.0,
        1,
        1,
        0.0
    )
    dchkp = Enzyme.make_zero(chkp)

    autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        exp2_cpintegrate,
        Active,
        Duplicated(chkp, dchkp),
        Const(revolve)
    )

    enzyme_deriv = dchkp.S.Prog.u[x_coord, y_coord]

    steps = [50, 40, 30, 20, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    P2 = ShallowWaters.Parameter(T=Float32;kwargs...)
    S1 = ShallowWaters.model_setup(P2)
    chkp1 = exp2_Chkp{T, T}(S1,
    data,
    data_spots,
    data_steps,
    0.0,
    1,
    1,
    0.0
    )
    J_outer = exp2_integrate(chkp1)

    diffs = []

    for s in steps

        P = ShallowWaters.Parameter(T=Float32;kwargs...)
        S_inner = ShallowWaters.model_setup(P)

        S_inner.Prog.u[x_coord, y_coord] += s

        chkp_inner = exp2_Chkp{T, T}(S_inner,
        data,
        data_spots,
        data_steps,
        0.0,
        1,
        1,
        0.0
        )

        revolve = Revolve{exp2_Chkp{T, T}}(chkp_inner.S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
        )

        J_inner = exp2_cpintegrate(chkp_inner, revolve)

        push!(diffs, (J_inner - J_outer) / s)

    end

    return diffs, enzyme_deriv

end

# xu = 30:10:100
# yu = 40:10:100
# Xu = xu' .* ones(length(yu))
# Yu = ones(length(xu))' .* yu

# sigma_data = 0.01
# sigma_initcond = 0.02
# data_steps = 220:220:6733
# data_spotsu = vec((Xu.-1) .* 127 + Yu)
# data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
# data_spots = [data_spotsu; data_spotsv]
# Ndays = 1

# diffs, enzyme_deriv = exp2_finite_difference_only(data_spots, 50, 50;
#     output=false,
#     L_ratio=1,
#     g=9.81,
#     H=500,
#     wind_forcing_x="double_gyre",
#     Lx=3840e3,
#     tracer_advection=false,
#     tracer_relaxation=false,
#     seasonal_wind_x=false,
#     data_steps=data_steps,
#     topography="flat",
#     bc="nonperiodic",
#     α=2,
#     nx=128,
#     Ndays=Ndays,
#     initial_cond="ncfile",
#     initpath="./data_files_forkf/128_spinup_noforcing/"
# )