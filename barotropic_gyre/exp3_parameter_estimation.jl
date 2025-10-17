mutable struct exp3_adj_model{T, S} <: AbstractNLPModel{T,S}
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

mutable struct exp3_ekf_model{T}
    S::ShallowWaters.ModelSetup{T,T}        # model struct for adjoint
    N::Int                                  # number of ensemble members
    data::Array{T, 2}                       # data to be assimilated
    sigma_initcond::T
    sigma_data::T
    data_steps::StepRange{Int, Int}         # when data is assimilated
    data_spots::Array{Int, 1}               # where data is located, grid coordinates
    j::Int                                  # for keeping track of location in data
end

function exp3_model_setup(T, Ndays, N, sigma_data, sigma_initcond, data_steps, data_spots)

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

    data, true_states = exp2_generate_data(S_true, data_steps, data_spots, sigma_data)

    S_pred = ShallowWaters.model_setup(P_pred)

    # perturb wind stress amplitude
    S_pred.parameters.Fx0 = .0001 * S_true.parameters.Fx0

    Prog_pred = ShallowWaters.PrognosticVars{T}(ShallowWaters.remove_halo(S_pred.Prog.u,
        S_pred.Prog.v,
        S_pred.Prog.η,
        S_pred.Prog.sst,
        S_pred)...
    )

    # perturb initial conditions from those seen by the "true" model (create incorrect initial conditions)
    # trying to do this by perturbing the high wavelengths instead of doing a smooth perturbation with one 
    # standard deviation
    # I'm (somewhat arbitrarily) choosing 1 ≤ n, m ≤ 20 for the wavenumbers
    upert = zeros(size(Prog_pred.u))
    vpert = zeros(size(Prog_pred.v))
    etapert = zeros(size(Prog_pred.η))

    for n = 1:5
        for m = 1:5
            urand = randn(4)
            vrand = randn(4)
            for k = 1:127
                for j = 1:128
                    upert[k,j] += sigma_initcond * urand[1] * cos((pi * n / 127) * k)*cos(pi * m / 128 * j)
                        + sigma_initcond * urand[2] * sin((pi * n / 127) * k)*cos(pi * m / 128 * j)
                        + sigma_initcond * urand[3] * cos((pi * n / 127) * k)*sin(pi * m / 128 * j)
                        + sigma_initcond * urand[4] * sin((pi * n / 127) * k)*sin(pi * m / 128 * j)
                    vpert[j,k] += sigma_initcond * vrand[1] * cos(pi * n / 128 * j) * cos(pi * m / 127 * k)
                        + sigma_initcond * vrand[2] * cos(pi * n / 128 * j) * sin(pi * m / 127 * k)
                        + sigma_initcond * vrand[3] * sin(pi * n / 128 * j) * cos(pi * m / 127 * k)
                        + sigma_initcond * vrand[4] * sin(pi * n / 128 * j) * sin(pi * m / 127 * k)
                end
            end

        end
    end

    for n = 1:5
        for m = 1:5
            etarand = randn(4)
            for k = 1:128
                for j = 1:128
                    etapert[k,j] += sigma_initcond * etarand[1] * cos((pi * n / 128) * k)*cos(pi * m / 128 * j)
                        + sigma_initcond * etarand[2] * cos((pi * n / 128) * k)*sin(pi * m / 128 * j)
                        + sigma_initcond * etarand[3] * sin((pi * n / 128) * k)*cos(pi * m / 128 * j)
                        + sigma_initcond * etarand[4] * sin((pi * n / 128) * k)*sin(pi * m / 128 * j)
                end
            end
        end
    end

    Prog_pred.u = Prog_pred.u + upert
    Prog_pred.v = Prog_pred.v + vpert
    Prog_pred.η = Prog_pred.η + etapert
    
    uic,vic,etaic = ShallowWaters.add_halo(Prog_pred.u,Prog_pred.v,Prog_pred.η,Prog_pred.sst,S_pred)
    S_pred.Prog.u = uic
    S_pred.Prog.v = vic
    S_pred.Prog.η = etaic

    param_guess = [vec(uic); vec(vic); vec(etaic)]

    Skf = deepcopy(S_pred)
    Sadj = deepcopy(S_pred)

    meta = NLPModelMeta(length(param_guess); ncon=0, nnzh=0,x0=param_guess)
    counters = Counters()

    adj_model = exp3_adj_model{T, typeof(param_guess)}(meta,
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

    ekf_model = exp3_ekf_model{T}(
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

function exp3_cpintegrate(chkp, scheme)::Float64

    # additions to get derivatives with respect to forcing amplitude
    forcing = chkp.S.forcing
    Fx, _ = ShallowWaters.DoubleGyreWind(typeof(forcing).parameters[1], chkp.S.parameters, chkp.S.grid)
    chkp.S.forcing = ShallowWaters.Forcing(Fx, forcing.Fy, forcing.H, forcing.η_ref, forcing.Fη)

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(chkp.S.Diag.VolumeFluxes.h, chkp.S.Prog.η, chkp.S.forcing.H)
    ShallowWaters.Ix!(chkp.S.Diag.VolumeFluxes.h_u, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(chkp.S.Diag.VolumeFluxes.h_v, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(chkp.S.Diag.Vorticity.h_q, chkp.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = Diag.PrognosticVarsRHS.u .= S.Prog.u
    vrhs = Diag.PrognosticVarsRHS.v .= S.Prog.v
    ηrhs = Diag.PrognosticVarsRHS.η .= S.Prog.η

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
            u1rhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Diag.RungeKutta.u1
            v1rhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Diag.RungeKutta.v1
            η1rhs = chkp.S.Diag.PrognosticVarsRHS.η .= chkp.S.Diag.RungeKutta.η1

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

        u0rhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Diag.RungeKutta.u0
        v0rhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Diag.RungeKutta.v0
        η0rhs = chkp.S.Diag.PrognosticVarsRHS.η .= chkp.S.Diag.RungeKutta.η0

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

    u0rhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Diag.RungeKutta.u0
    v0rhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Diag.RungeKutta.v0
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

        chkp.J += abs(chkp.S.parameters.Fx0)^2

        chkp.j += 1
    end

    copyto!(chkp.S.Prog.u, chkp.S.Diag.RungeKutta.u0)
    copyto!(chkp.S.Prog.v, chkp.S.Diag.RungeKutta.v0)
    copyto!(chkp.S.Prog.η, chkp.S.Diag.RungeKutta.η0)
    end

    return chkp.J

end

function exp3_integrate(model)

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
        temp = ShallowWaters.PrognosticVars{model.S.parameters.T}(ShallowWaters.remove_halo(
            model.S.Prog.u,
            model.S.Prog.v,
            model.S.Prog.η,
            model.S.Prog.sst,
            model.S
        )...)

        tempuv = [vec(temp.u); vec(temp.v)][model.data_spots]

        model.J += sum((tempuv - model.data[:, model.j]).^2)

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

    S = model.S
    current = 1
    for m in (S.Prog.u, S.Prog.v, S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    S.parameters.Fx0 = param_guess[end]
    data_spots = Int.(data_spots)

    return exp3_integrate(model)

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

    S = model.S
    current = 1
    for m in (S.Prog.u, S.Prog.v, S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    S.parameters.Fx0 = param_guess[end]

    snaps = Int(floor(sqrt(model.S.grid.nt)))
    revolve = Revolve(
        snaps;
        verbose=0,
        gc=true,
        write_checkpoints=false
    )

    dmodel = Enzyme.make_zero(model)

    @time J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        exp3_cpintegrate,
        Active,
        Duplicated(model, dmodel),
        Const(revolve)
    )[2]

    # Get gradient
    G .= [vec(dmodel.S.Prog.u); vec(dmodel.S.Prog.v); vec(dmodel.S.Prog.η); dchkp.S.parameters.Fx0]

    return nothing

end

function run_exp3()

    Ndays = 30
    N = 10
    sigma_data = 0.5        # this and the init cond are what the result from optimization used, be careful in adjusting
    sigma_initcond = 10.
    data_steps = 225:224:6733
    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]

    adj_model, ekf_model, param_guess, S_pred, P_pred, true_states = exp3_model_setup(Float64, Ndays,N,sigma_data,sigma_initcond,data_steps,data_spots)

    # run the ensemble Kalman filter
    ekf_avgu, ekf_avgv = run_ensemble_kf(ekf_model, param_guess)

    # run the adjoint optimization
    qn_options = MadNLP.QuasiNewtonOptions(; max_history=200)
    result = madnlp(
        adj_model;
        # linear_solver=LapackCPUSolver,
        hessian_approximation=MadNLP.CompactLBFGS,
        quasi_newton_options=qn_options
        # max_iter=1000
    )

    S_adj = ShallowWaters.model_setup(P_pred)
    current = 1
    for m in (S_adj.Prog.u, S_adj.Prog.v, S_adj.Prog.η)
        sz = prod(size(m))
        m .= reshape(result.solution[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    S_adj.parameters.Fx0 = result.solution[end]

    _, states_adj = exp1_generate_data(S_adj, data_steps, data_spots, sigma_data)

    return result, pred_states, ekf_avgu, ekf_avgv, data, true_states, S_adj, states_adj, S_pred

end

function exp3_plots()

    # S_kf_all, Progkf_all, G, dS, data, true_states, result, S_adj, states_adj

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    N = 10
    sigma_data = 0.01
    sigma_initcond = 0.02
    data_steps = 220:220:6733
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
    t = 673
    ax1, hm1 = heatmap(fig1[1,1], states_adj[t].v,
    colormap=:balance,
    colorrange=(-maximum(states_adj[t].v),
    maximum(states_adj[t].v)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y, +)")
    );
    Colorbar(fig1[1,2], hm1)
    ax2, hm2 = heatmap(fig1[1,3], abs.(true_states[t].v .- states_adj[t].v),
    colormap=:amp,
    colorrange=(0,
    1.5),
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
    colorrange=(0,
    1.5),
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

end

function exp3_finitedifference(x_coord, y_coord)

    Ndays = 10
    T = Float32
    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    sigma_data = 0.01
    data_steps = 1:200
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)
    data_spots = [data_spotsu; data_spotsv]

    P1 = ShallowWaters.Parameter(T=Float32;
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

    S_true = ShallowWaters.model_setup(P1)
    data, _ = exp3_generate_data(S_true, data_spots, sigma_data)

    snaps = Int(floor(sqrt(S_true.grid.nt)))
    revolve = Revolve{ShallowWaters.ModelSetup}(S_true.grid.nt,
        snaps;
        verbose=1,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 224
    )

    P2 = ShallowWaters.Parameter(T=T;
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
        Fx0 = 0.00012,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    S_pred = ShallowWaters.model_setup(P2)

    chkp = exp3_Chkp{T, T}(S_pred,
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
    J_outer = exp3_cpintegrate(chkp_prim, revolve)
    println("Cost without AD: $J_outer")

    @time J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        exp3_cpintegrate,
        Active,
        Duplicated(chkp, dchkp),
        Const(revolve)
    )[2]
    println("Cost with AD: $J")

    enzyme_deriv = dchkp.S.parameters.Fx0

    steps = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    diffs = []

    for s in steps

        P3 = ShallowWaters.Parameter(T=Float32;output=false,
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
        Fx0 = 0.00012 + s,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/")

        S_inner = ShallowWaters.model_setup(P3)

        chkp_inner = exp3_Chkp{T, T}(S_inner,
            data,
            data_spots,
            data_steps,
            0.0,
            1,
            1,
            0.0
        )

        J_inner = exp3_cpintegrate(chkp_inner, revolve)

        push!(diffs, (J_inner - J_outer) / s)

    end

    return diffs, enzyme_deriv, chkp_prim, dchkp

end