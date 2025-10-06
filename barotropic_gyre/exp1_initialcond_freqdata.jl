mutable struct exp1_adj_model{T, S} <: AbstractNLPModel{T,S}
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

mutable struct exp1_ekf_model{T}
    S::ShallowWaters.ModelSetup{T,T}        # model struct for adjoint
    N::Int                                  # number of ensemble members
    data::Array{T, 2}                       # data to be assimilated
    sigma_initcond::T
    sigma_data::T
    data_steps::StepRange{Int, Int}         # when data is assimilated
    data_spots::Array{Int, 1}               # where data is located, grid coordinates
    j::Int                                  # for keeping track of location in data
end

function exp1_model_setup(T, Ndays, N, sigma_data, sigma_initcond, data_steps, data_spots)

    P_pred = ShallowWaters.Parameter(T=T;
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
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    S_true = ShallowWaters.model_setup(P_pred)

    data, true_states = exp1_generate_data(S_true, data_steps, data_spots, sigma_data)

    S_pred = ShallowWaters.model_setup(P_pred)

    Prog_pred = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S_pred.Prog.u,
        S_pred.Prog.v,
        S_pred.Prog.η,
        S_pred.Prog.sst,
        S_pred)...
    )

    # perturb initial conditions from those seen by the "true" model (create incorrect initial conditions)
    Prog_pred.u = Prog_pred.u + sigma_initcond .* randn(size(Prog_pred.u))
    Prog_pred.v = Prog_pred.v + sigma_initcond .* randn(size(Prog_pred.v))
    Prog_pred.η = Prog_pred.η + sigma_initcond .* randn(size(Prog_pred.η))
    
    uic,vic,etaic = ShallowWaters.add_halo(Prog_pred.u,Prog_pred.v,Prog_pred.η,Prog_pred.sst,S_pred)
    S_pred.Prog.u = uic
    S_pred.Prog.v = vic
    S_pred.Prog.η = etaic

    param_guess = [vec(uic); vec(vic); vec(etaic)]

    Skf = deepcopy(S_pred)
    Sadj = deepcopy(S_pred)

    meta = NLPModelMeta(length(param_guess); ncon=0, nnzh=0,x0=param_guess)
    counters = Counters()

    adj_model = exp1_adj_model{T, typeof(param_guess)}(meta,
        Counters(),
        Sadj,
        data,
        data_steps,
        data_spots,
        0.0,
        1,
        1,
        0
    )

    ekf_model = exp1_ekf_model{T}(
        Skf,
        N,
        data,
        sigma_initcond,
        sigma_data,
        data_steps,
        data_spots,
        1
    )


    return adj_model, ekf_model, param_guess, S_pred, P_pred, true_states
end

function exp1_cpintegrate(model, scheme)

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(model.S.Diag.VolumeFluxes.h, model.S.Prog.η, model.S.forcing.H)
    ShallowWaters.Ix!(model.S.Diag.VolumeFluxes.h_u, model.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(model.S.Diag.VolumeFluxes.h_v, model.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(model.S.Diag.Vorticity.h_q, model.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Prog.u
    vrhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Prog.v
    ηrhs = model.S.Diag.PrognosticVarsRHS.η .= model.S.Prog.η

    ShallowWaters.advection_coriolis!(urhs, vrhs, ηrhs, model.S.Diag, model.S)
    ShallowWaters.PVadvection!(model.S.Diag, model.S)

    # propagate initial conditions
    copyto!(model.S.Diag.RungeKutta.u0, model.S.Prog.u)
    copyto!(model.S.Diag.RungeKutta.v0, model.S.Prog.v)
    copyto!(model.S.Diag.RungeKutta.η0, model.S.Prog.η)

    # store initial conditions of sst for relaxation
    copyto!(model.S.Diag.SemiLagrange.sst_ref, model.S.Prog.sst)

    # run integration loop with checkpointing
    model.j = 1
    @ad_checkpoint scheme for model.i = 1:model.S.grid.nt

        t = model.S.t
        i = model.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(model.S.Prog.u, model.S.Prog.v, model.S.Prog.η, model.S)
        copyto!(model.S.Diag.RungeKutta.u1, model.S.Prog.u)
        copyto!(model.S.Diag.RungeKutta.v1, model.S.Prog.v)
        copyto!(model.S.Diag.RungeKutta.η1, model.S.Prog.η)

        if model.S.parameters.compensated
            fill!(model.S.Diag.Tendencies.du_sum, zero(model.S.parameters.Tprog))
            fill!(model.S.Diag.Tendencies.dv_sum, zero(model.S.parameters.Tprog))
            fill!(model.S.Diag.Tendencies.dη_sum, zero(model.S.parameters.Tprog))
        end

        for rki = 1:model.S.parameters.RKo
            if rki > 1
                ShallowWaters.ghost_points!(
                    model.S.Diag.RungeKutta.u1,
                    model.S.Diag.RungeKutta.v1,
                    model.S.Diag.RungeKutta.η1,
                    model.S
                )
            end

            # type conversion for mixed precision
            u1rhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Diag.RungeKutta.u1
            v1rhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Diag.RungeKutta.v1
            η1rhs = model.S.Diag.PrognosticVarsRHS.η .= model.S.Diag.RungeKutta.η1

            ShallowWaters.rhs!(u1rhs, v1rhs, η1rhs, model.S.Diag, model.S, t)          # momentum only
            ShallowWaters.continuity!(u1rhs, v1rhs, η1rhs, model.S.Diag, model.S, t)   # continuity equation

            if rki < model.S.parameters.RKo
                ShallowWaters.caxb!(
                    model.S.Diag.RungeKutta.u1,
                    model.S.Prog.u,
                    model.S.constants.RKbΔt[rki],
                    model.S.Diag.Tendencies.du
                )
                ShallowWaters.caxb!(
                    model.S.Diag.RungeKutta.v1,
                    model.S.Prog.v,
                    model.S.constants.RKbΔt[rki],
                    model.S.Diag.Tendencies.dv
                )
                ShallowWaters.caxb!(
                    model.S.Diag.RungeKutta.η1,
                    model.S.Prog.η,
                    model.S.constants.RKbΔt[rki],
                    model.S.Diag.Tendencies.dη
                )
            end

            if model.S.parameters.compensated
                ShallowWaters.axb!(model.S.Diag.Tendencies.du_sum, model.S.constants.RKaΔt[rki], model.S.Diag.Tendencies.du)
                ShallowWaters.axb!(model.S.Diag.Tendencies.dv_sum, model.S.constants.RKaΔt[rki], model.S.Diag.Tendencies.dv)
                ShallowWaters.axb!(model.S.Diag.Tendencies.dη_sum, model.S.constants.RKaΔt[rki], model.S.Diag.Tendencies.dη)
            else
                ShallowWaters.axb!(
                    model.S.Diag.RungeKutta.u0,
                    model.S.constants.RKaΔt[rki],
                    model.S.Diag.Tendencies.du
                )
                ShallowWaters.axb!(
                    model.S.Diag.RungeKutta.v0,
                    model.S.constants.RKaΔt[rki],
                    model.S.Diag.Tendencies.dv
                )
                ShallowWaters.axb!(
                    model.S.Diag.RungeKutta.η0,
                    model.S.constants.RKaΔt[rki],
                    model.S.Diag.Tendencies.dη
                )
            end
        end

        if model.S.parameters.compensated
            ShallowWaters.axb!(model.S.Diag.Tendencies.du_sum, -1, model.S.Diag.Tendencies.du_comp)
            ShallowWaters.axb!(model.S.Diag.Tendencies.dv_sum, -1, model.S.Diag.Tendencies.dv_comp)
            ShallowWaters.axb!(model.S.Diag.Tendencies.dη_sum, -1, model.S.Diag.Tendencies.dη_comp)

            ShallowWaters.axb!(model.S.Diag.RungeKutta.u0, 1, model.S.Diag.Tendencies.du_sum)
            ShallowWaters.axb!(model.S.Diag.RungeKutta.v0, 1, model.S.Diag.Tendencies.dv_sum)
            ShallowWaters.axb!(model.S.Diag.RungeKutta.η0, 1, model.S.Diag.Tendencies.dη_sum)

            ShallowWaters.dambmc!(
                model.S.Diag.Tendencies.du_comp,
                model.S.Diag.RungeKutta.u0,
                model.S.Prog.u,
                model.S.Diag.Tendencies.du_sum
            )
            ShallowWaters.dambmc!(
                model.S.Diag.Tendencies.dv_comp,
                model.S.Diag.RungeKutta.v0,
                model.S.Prog.v,
                model.S.Diag.Tendencies.dv_sum
            )
            ShallowWaters.dambmc!(
                model.S.Diag.Tendencies.dη_comp,
                model.S.Diag.RungeKutta.η0,
                model.S.Prog.η,
                model.S.Diag.Tendencies.dη_sum
            )
        end

        ShallowWaters.ghost_points!(
            model.S.Diag.RungeKutta.u0,
            model.S.Diag.RungeKutta.v0,
            model.S.Diag.RungeKutta.η0,
            model.S
        )

        u0rhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Diag.RungeKutta.u0
        v0rhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Diag.RungeKutta.v0
        η0rhs = model.S.Diag.PrognosticVarsRHS.η .= model.S.Diag.RungeKutta.η0

        if model.S.parameters.dynamics == "nonlinear" && model.S.grid.nstep_advcor > 0 && (i % model.S.grid.nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs, v0rhs, η0rhs, model.S.Diag, model.S)
            ShallowWaters.advection_coriolis!(u0rhs, v0rhs, η0rhs, model.S.Diag, model.S)
        end

        if (model.S.parameters.i % model.S.grid.nstep_diff) == 0
        ShallowWaters.bottom_drag!(u0rhs, v0rhs, η0rhs, model.S.Diag, model.S)
        ShallowWaters.diffusion!(u0rhs, v0rhs, model.S.Diag, model.S)
        ShallowWaters.add_drag_diff_tendencies!(
            model.S.Diag.RungeKutta.u0,
            model.S.Diag.RungeKutta.v0,
            model.S.Diag,
            model.S
        )
        ShallowWaters.ghost_points_uv!(
            model.S.Diag.RungeKutta.u0,
            model.S.Diag.RungeKutta.v0,
            model.S
        )
    end

    t += model.S.grid.dtint

    u0rhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Diag.RungeKutta.u0
    v0rhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Diag.RungeKutta.v0
    ShallowWaters.tracer!(i, u0rhs, v0rhs, model.S.Prog, model.S.Diag, model.S)

    if i in model.data_steps

        temp = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(model.S.Prog.u,
        model.S.Prog.v,
        model.S.Prog.η,
        model.S.Prog.sst,model.S)...)

        tempuv = [vec(temp.u);vec(temp.v)]

        model.J += sum((tempuv[model.data_spots] - model.data[:, model.j]).^2)

        model.j += 1

    end

    copyto!(model.S.Prog.u, model.S.Diag.RungeKutta.u0)
    copyto!(model.S.Prog.v, model.S.Diag.RungeKutta.v0)
    copyto!(model.S.Prog.η, model.S.Diag.RungeKutta.η0)
    end

    return model.J

end

function exp1_integrate(model)

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(model.S.Diag.VolumeFluxes.h, model.S.Prog.η, model.S.forcing.H)
    ShallowWaters.Ix!(model.S.Diag.VolumeFluxes.h_u, model.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(model.S.Diag.VolumeFluxes.h_v, model.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(model.S.Diag.Vorticity.h_q, model.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Prog.u
    vrhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Prog.v
    ηrhs = model.S.Diag.PrognosticVarsRHS.η .= model.S.Prog.η

    ShallowWaters.advection_coriolis!(urhs, vrhs, ηrhs, model.S.Diag, model.S)
    ShallowWaters.PVadvection!(model.S.Diag, model.S)

    # propagate initial conditions
    copyto!(model.S.Diag.RungeKutta.u0, model.S.Prog.u)
    copyto!(model.S.Diag.RungeKutta.v0, model.S.Prog.v)
    copyto!(model.S.Diag.RungeKutta.η0, model.S.Prog.η)

    # store initial conditions of sst for relaxation
    copyto!(model.S.Diag.SemiLagrange.sst_ref, model.S.Prog.sst)

    # run integration loop with checkpointing
    model.j = 1
    for model.i = 1:model.S.grid.nt

        t = model.S.t
        i = model.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(model.S.Prog.u, model.S.Prog.v, model.S.Prog.η, model.S)
        copyto!(model.S.Diag.RungeKutta.u1, model.S.Prog.u)
        copyto!(model.S.Diag.RungeKutta.v1, model.S.Prog.v)
        copyto!(model.S.Diag.RungeKutta.η1, model.S.Prog.η)

        if model.S.parameters.compensated
            fill!(model.S.Diag.Tendencies.du_sum, zero(model.S.parameters.Tprog))
            fill!(model.S.Diag.Tendencies.dv_sum, zero(model.S.parameters.Tprog))
            fill!(model.S.Diag.Tendencies.dη_sum, zero(model.S.parameters.Tprog))
        end

        for rki = 1:model.S.parameters.RKo
            if rki > 1
                ShallowWaters.ghost_points!(
                    model.S.Diag.RungeKutta.u1,
                    model.S.Diag.RungeKutta.v1,
                    model.S.Diag.RungeKutta.η1,
                    model.S
                )
            end

            # type conversion for mixed precision
            u1rhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Diag.RungeKutta.u1
            v1rhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Diag.RungeKutta.v1
            η1rhs = model.S.Diag.PrognosticVarsRHS.η .= model.S.Diag.RungeKutta.η1

            ShallowWaters.rhs!(u1rhs, v1rhs, η1rhs, model.S.Diag, model.S, t)          # momentum only
            ShallowWaters.continuity!(u1rhs, v1rhs, η1rhs, model.S.Diag, model.S, t)   # continuity equation

            if rki < model.S.parameters.RKo
                ShallowWaters.caxb!(
                    model.S.Diag.RungeKutta.u1,
                    model.S.Prog.u,
                    model.S.constants.RKbΔt[rki],
                    model.S.Diag.Tendencies.du
                )
                ShallowWaters.caxb!(
                    model.S.Diag.RungeKutta.v1,
                    model.S.Prog.v,
                    model.S.constants.RKbΔt[rki],
                    model.S.Diag.Tendencies.dv
                )
                ShallowWaters.caxb!(
                    model.S.Diag.RungeKutta.η1,
                    model.S.Prog.η,
                    model.S.constants.RKbΔt[rki],
                    model.S.Diag.Tendencies.dη
                )
            end

            if model.S.parameters.compensated
                ShallowWaters.axb!(model.S.Diag.Tendencies.du_sum, model.S.constants.RKaΔt[rki], model.S.Diag.Tendencies.du)
                ShallowWaters.axb!(model.S.Diag.Tendencies.dv_sum, model.S.constants.RKaΔt[rki], model.S.Diag.Tendencies.dv)
                ShallowWaters.axb!(model.S.Diag.Tendencies.dη_sum, model.S.constants.RKaΔt[rki], model.S.Diag.Tendencies.dη)
            else
                ShallowWaters.axb!(
                    model.S.Diag.RungeKutta.u0,
                    model.S.constants.RKaΔt[rki],
                    model.S.Diag.Tendencies.du
                )
                ShallowWaters.axb!(
                    model.S.Diag.RungeKutta.v0,
                    model.S.constants.RKaΔt[rki],
                    model.S.Diag.Tendencies.dv
                )
                ShallowWaters.axb!(
                    model.S.Diag.RungeKutta.η0,
                    model.S.constants.RKaΔt[rki],
                    model.S.Diag.Tendencies.dη
                )
            end
        end

        if model.S.parameters.compensated
            ShallowWaters.axb!(model.S.Diag.Tendencies.du_sum, -1, model.S.Diag.Tendencies.du_comp)
            ShallowWaters.axb!(model.S.Diag.Tendencies.dv_sum, -1, model.S.Diag.Tendencies.dv_comp)
            ShallowWaters.axb!(model.S.Diag.Tendencies.dη_sum, -1, model.S.Diag.Tendencies.dη_comp)

            ShallowWaters.axb!(model.S.Diag.RungeKutta.u0, 1, model.S.Diag.Tendencies.du_sum)
            ShallowWaters.axb!(model.S.Diag.RungeKutta.v0, 1, model.S.Diag.Tendencies.dv_sum)
            ShallowWaters.axb!(model.S.Diag.RungeKutta.η0, 1, model.S.Diag.Tendencies.dη_sum)

            ShallowWaters.dambmc!(
                model.S.Diag.Tendencies.du_comp,
                model.S.Diag.RungeKutta.u0,
                model.S.Prog.u,
                model.S.Diag.Tendencies.du_sum
            )
            ShallowWaters.dambmc!(
                model.S.Diag.Tendencies.dv_comp,
                model.S.Diag.RungeKutta.v0,
                model.S.Prog.v,
                model.S.Diag.Tendencies.dv_sum
            )
            ShallowWaters.dambmc!(
                model.S.Diag.Tendencies.dη_comp,
                model.S.Diag.RungeKutta.η0,
                model.S.Prog.η,
                model.S.Diag.Tendencies.dη_sum
            )
        end

        ShallowWaters.ghost_points!(
            model.S.Diag.RungeKutta.u0,
            model.S.Diag.RungeKutta.v0,
            model.S.Diag.RungeKutta.η0,
            model.S
        )

        u0rhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Diag.RungeKutta.u0
        v0rhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Diag.RungeKutta.v0
        η0rhs = model.S.Diag.PrognosticVarsRHS.η .= model.S.Diag.RungeKutta.η0

        if model.S.parameters.dynamics == "nonlinear" && model.S.grid.nstep_advcor > 0 && (i % model.S.grid.nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs, v0rhs, η0rhs, model.S.Diag, model.S)
            ShallowWaters.advection_coriolis!(u0rhs, v0rhs, η0rhs, model.S.Diag, model.S)
        end

        if (model.S.parameters.i % model.S.grid.nstep_diff) == 0
        ShallowWaters.bottom_drag!(u0rhs, v0rhs, η0rhs, model.S.Diag, model.S)
        ShallowWaters.diffusion!(u0rhs, v0rhs, model.S.Diag, model.S)
        ShallowWaters.add_drag_diff_tendencies!(
            model.S.Diag.RungeKutta.u0,
            model.S.Diag.RungeKutta.v0,
            model.S.Diag,
            model.S
        )
        ShallowWaters.ghost_points_uv!(
            model.S.Diag.RungeKutta.u0,
            model.S.Diag.RungeKutta.v0,
            model.S
        )
    end

    t += model.S.grid.dtint

    u0rhs = model.S.Diag.PrognosticVarsRHS.u .= model.S.Diag.RungeKutta.u0
    v0rhs = model.S.Diag.PrognosticVarsRHS.v .= model.S.Diag.RungeKutta.v0
    ShallowWaters.tracer!(i, u0rhs, v0rhs, model.S.Prog, model.S.Diag, model.S)

    if i in model.data_steps

        temp = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(model.S.Prog.u,
        model.S.Prog.v,
        model.S.Prog.η,
        model.S.Prog.sst,model.S)...)

        tempuv = [vec(temp.u);vec(temp.v)]

        model.J += sum((tempuv[model.data_spots] - model.data[:, model.j]).^2)

        model.j += 1

    end

    copyto!(model.S.Prog.u, model.S.Diag.RungeKutta.u0)
    copyto!(model.S.Prog.v, model.S.Diag.RungeKutta.v0)
    copyto!(model.S.Prog.η, model.S.Diag.RungeKutta.η0)
    end

    return model.J

end

function NLPModels.obj(model, param_guess)

    P = ShallowWaters.Parameter(T=model.S.parameters.T;
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
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )
    model.S = ShallowWaters.model_setup(P)
    model.J = 0.0
    model.j = 1
    model.i = 1
    model.t = 0

    current = 1
    for m in (model.S.Prog.u, model.S.Prog.v, model.S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    return exp1_integrate(model)

end

function NLPModels.grad!(model, param_guess, G)

    P = ShallowWaters.Parameter(T=model.S.parameters.T;
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
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )
    model.S = ShallowWaters.model_setup(P)
    model.J = 0.0
    model.t = 0.0
    model.j = 1

    current = 1
    for m in (model.S.Prog.u, model.S.Prog.v, model.S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    # if we want to use checkpointing
    snaps = Int(floor(sqrt(model.S.grid.nt)))
    revolve = Revolve{exp1_adj_model}(model.S.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )

    dmodel = Enzyme.make_zero(model)

    J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        exp1_integrate,
        Active,
        Duplicated(model, dmodel)
    )[2]

    # derivative of loss with respect to initial condition
    G .= [vec(dmodel.S.Prog.u); vec(dmodel.S.Prog.v); vec(dmodel.S.Prog.η)]

    return nothing

end

function run_exp1()

    Ndays = 10
    N = 10
    sigma_data = 0.01
    sigma_initcond = 0.02
    data_steps = 220:220:6733

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)
    data_spots = [data_spotsu; data_spotsv]

    # setup all models
    adj_model, ekf_model, param_guess, S_pred, P_pred, true_states = exp1_model_setup(Float64, Ndays,N,sigma_data,sigma_initcond,data_steps,data_spots)

    # run the ensemble Kalman filter
    ekf_avgu, ekf_avgv = run_ensemble_kf(ekf_model, param_guess)

    # run the adjoint optimization
    qn_options = MadNLP.QuasiNewtonOptions(; max_history=200)
    result = madnlp(
        nlp;
        # linear_solver=LapackCPUSolver,
        hessian_approximation=MadNLP.CompactLBFGS,
        quasi_newton_options=qn_options
        # max_iter=1000
    )

    # integrate with the result from the optimization
    S_adj = ShallowWaters.model_setup(P_pred)
    current = 1
    for m in (S_adj.Prog.u, S_adj.Prog.v, S_adj.Prog.η)
        sz = prod(size(m))
        m .= reshape(result.solution[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    _, states_adj = exp1_generate_data(S_adj, data_steps, data_spots, sigma_data)

    return ekf_avgu, ekf_avgv, G, dS, data, true_states, result, S_adj, states_adj

end

function exp1_plots()

    # S_kf_all, Progkf_all, G, dS, data, states_true, result, S_adj, states_adj

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    N = 10
    sigma_data = 0.01
    sigma_initcond = 0.02
    data_steps = 200:200:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spots = [data_spotsu; data_spotsv]
    Ndays = 30

    fig1 = Figure(size=(600, 500));
    ax1, hm1 = heatmap(fig1[1,1], states_true[end].u,
        colormap=:balance,
        colorrange=(-maximum(states_true[end].u),
        maximum(states_true[end].u)),
        axis=(xlabel=L"x", ylabel=L"y", title=L"u(t = 30 \; \text{days}, x, y)")
    );
    scatter!(ax1, vec(X), vec(Y), color=:green);
    Colorbar(fig1[1,2], hm1)
    fig1

    # fig1, ax1, hm1 = heatmap(states_true[end].u,
    # colormap=:balance,
    # colorrange=(-maximum(states_true[end].u),
    # maximum(states_true[end].u)),
    # axis=(xlabel=L"x", ylabel=L"y", title=L"u(t = 30 \; \text{days}, x, y)")
    # );
    # Colorbar(fig1[1,2], hm1);

    # u and v plot

    kf_avgu = zeros(127,128)
    kf_avgv = zeros(128,127)
    for n = 1:10
        kf_avgu = kf_avgu .+ Progkf_all[end][n].u
        kf_avgv = kf_avgv .+ Progkf_all[end][n].v
    end
    kf_avgu = kf_avgu ./ 10
    kf_avgv = kf_avgv ./ 10

    fig1 = Figure(size=(800,700));
    ax1, hm1 = heatmap(fig1[1,1], states_adj[end].u,
    colormap=:balance,
    colorrange=(-maximum(states_true[end].u),
    maximum(states_true[end].u)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{u}(t = 30 \; \text{days}, x, y, +)")
    );
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], abs.(states_true[end].u .- states_adj[end].u),
    colormap=:amp,
    colorrange=(0,
    1.5),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|u(x,y) - \tilde{u}(x, y, +)|")
    )
    Colorbar(fig1[1,4], hm2)

    ax3, hm3 = heatmap(fig1[2, 1], kf_avgu,
    colormap=:balance,
    colorrange=(-maximum(states_true[end].u),
    maximum(states_true[end].u)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{u}(t = 30 \; \text{days}, x, y)")
    );
    Colorbar(fig1[2,2], hm3)
    ax4, hm4 = heatmap(fig1[2,3], abs.(states_true[end].u .- kf_avgu),
    colormap=:amp,
    colorrange=(0,
    1.5),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|u(x,y) - \tilde{u}(x, y)|")
    )
    Colorbar(fig1[2,4], hm4)

    # energy plots

    fig1 = Figure(size=(600, 500))
    ax1, hm1 = heatmap(fig1[1,1], (states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2),
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\mathcal{E}"),
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2))
    );
    Colorbar(fig1[1,2], hm1)

    fig1 = Figure(size=(800, 700));
    ax1, hm1 = heatmap(fig1[1,1], states_adj[end].u[:, 1:end-1].^2 .+ states_adj[end].v[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}(+)"),
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    );
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], abs.((states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2) .- (states_adj[end].u[:, 1:end-1].^2 .+ states_adj[end].v[1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}(+)|")
    )
    Colorbar(fig1[1,4], hm2)


    ax3, hm3 = heatmap(fig1[2, 1], kf_avgu[:, 1:end-1].^2 .+ kf_avgv[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}"),
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    );
    Colorbar(fig1[2,2], hm3)
    ax4, hm4 = heatmap(fig1[2,3], abs.((states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2) .- (kf_avgu[:, 1:end-1].^2 .+ kf_avgv[1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(states_true[end].u[:, 1:end-1].^2 .+ states_true[end].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}|")
    )
    Colorbar(fig1[2,4], hm4)

end

########################################################################
# Derivative check

# function finite_difference_only(data_spots,x_coord,y_coord;kwargs...)

#     data, true_states = generate_data(data_spots, sigma_data; kwargs...)
#     P1 = ShallowWaters.Parameter(T=Float32;kwargs...)

#     S_outer = ShallowWaters.model_setup(P1)

#     dS_outer = Enzyme.Compiler.make_zero(S_outer)

#     ddata = Enzyme.make_zero(data)
#     ddata_spots = Enzyme.make_zero(data_spots)

#     autodiff(Enzyme.ReverseWithPrimal, integrate,
#         Duplicated(S_outer, dS_outer),
#         Duplicated(data,ddata),
#         Duplicated(data_spots, ddata_spots)
#     )

#     enzyme_deriv = dS_outer.Prog.u[x_coord, y_coord]

#     steps = [50, 40, 30, 20, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

#     P2 = ShallowWaters.Parameter(T=Float32;kwargs...)
#     S = ShallowWaters.model_setup(P2)
#     J_outer = integrate(S, data, data_spots)

#     diffs = []

#     for s in steps

#         P = ShallowWaters.Parameter(T=Float32;kwargs...)
#         S_inner = ShallowWaters.model_setup(P)

#         S_inner.Prog.u[x_coord, y_coord] += s

#         # J_inner = checkpointed_integration(S_inner, revolve)
#         J_inner = integrate(S_inner,data,data_spots)

#         push!(diffs, (J_inner - J_outer) / s)

#     end

#     return diffs, enzyme_deriv

# end

# diffs, enzyme_deriv = finite_difference_only(data_spots, 72, 64;
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