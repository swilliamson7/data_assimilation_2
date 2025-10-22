mutable struct exp2_adj_model{T, S} <: AbstractNLPModel{T,S}
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

mutable struct exp2_ekf_model{T}
    S::ShallowWaters.ModelSetup{T,T}        # model struct for adjoint
    N::Int                                  # number of ensemble members
    data::Array{T, 2}                       # data to be assimilated
    sigma_initcond::T
    sigma_data::T
    data_steps::StepRange{Int, Int}         # when data is assimilated
    data_spots::Array{Int, 1}               # where data is located, grid coordinates
    j::Int                                  # for keeping track of location in data
end

function exp2_model_setup(T, Ndays, N, sigma_data, sigma_initcond, data_steps, data_spots)

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

    Spred = deepcopy(S_pred)
    _, pred_states = exp2_generate_data(Spred, data_steps, data_spots, sigma_data)

    Skf = deepcopy(S_pred)
    Sadj = deepcopy(S_pred)

    meta = NLPModelMeta(length(param_guess); ncon=0, nnzh=0,x0=param_guess)
    counters = Counters()

    adj_model = exp2_adj_model{T, typeof(param_guess)}(meta,
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

    ekf_model = exp2_ekf_model{T}(
        Skf,
        N,
        data,
        sigma_initcond,
        sigma_data,
        data_steps,
        data_spots,
        1
    )
    return adj_model, ekf_model, param_guess, S_pred, pred_states, P_pred, true_states
end

function exp2_cpintegrate(chkp, scheme)::Float64

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(chkp.S.Diag.VolumeFluxes.h, chkp.S.Prog.η, chkp.S.forcing.H)
    ShallowWaters.Ix!(chkp.S.Diag.VolumeFluxes.h_u, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(chkp.S.Diag.VolumeFluxes.h_v, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(chkp.S.Diag.Vorticity.h_q, chkp.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Prog.u
    vrhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Prog.v
    ηrhs = chkp.S.Diag.PrognosticVarsRHS.η .= chkp.S.Prog.η

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
        temp = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
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

function exp2_integrate(chkp)::Float64

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(chkp.S.Diag.VolumeFluxes.h, chkp.S.Prog.η, chkp.S.forcing.H)
    ShallowWaters.Ix!(chkp.S.Diag.VolumeFluxes.h_u, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(chkp.S.Diag.VolumeFluxes.h_v, chkp.S.Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(chkp.S.Diag.Vorticity.h_q, chkp.S.Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = chkp.S.Diag.PrognosticVarsRHS.u .= chkp.S.Prog.u
    vrhs = chkp.S.Diag.PrognosticVarsRHS.v .= chkp.S.Prog.v
    ηrhs = chkp.S.Diag.PrognosticVarsRHS.η .= chkp.S.Prog.η

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
    for chkp.i = 1:chkp.S.grid.nt

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

    # if i === 1
        
    # end

    if i in chkp.data_steps
        temp = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(
            chkp.S.Prog.u,
            chkp.S.Prog.v,
            chkp.S.Prog.η,
            chkp.S.Prog.sst,
            chkp.S
        )...)

        tempuveta = [vec(temp.u); vec(temp.v)][model.data_spots]
        chkp.J += sum((tempuveta - chkp.data[:, chkp.j]).^2)

        chkp.j += 1
    end

    copyto!(chkp.S.Prog.u, chkp.S.Diag.RungeKutta.u0)
    copyto!(chkp.S.Prog.v, chkp.S.Diag.RungeKutta.v0)
    copyto!(chkp.S.Prog.η, chkp.S.Diag.RungeKutta.η0)
    end

    return chkp.J

end

function NLPModels.obj(model, param_guess)

    P_temp = ShallowWaters.Parameter(T=model.S.parameters.T;output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        tracer_advection=false,
        tracer_relaxation=false,
        α=2,
        nx=128,
        Ndays=model.S.parameters.Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/",
        init_starti=1
    )

    model.S = ShallowWaters.model_setup(P_temp)
    model.J = 0.0
    model.t = 0.0
    model.j = 1

    # modifying initial condition with the halo
    current = 1
    for m in (model.S.Prog.u, model.S.Prog.v)#, model.S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    # modifying initial condition with the halo
    # model.u_nohalo .= scale_inv*model.S.Prog.u[halo+1:end-halo,halo+1:end-halo]
    # model.v_nohalo .= scale_inv*model.S.Prog.v[halo+1:end-halo,halo+1:end-halo]
    # current = 1
    # for m in (model.u_nohalo, model.v_nohalo)#, model.S.Prog.η)
    #     sz = prod(size(m))
    #     m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
    #     current += sz
    # end

    # # add halo back for integrating
    # model.S.Prog.u .= scale*(cat(zeros(P_temp.T,halo,nuy+2*halo),cat(-model.u_nohalo[:,[2,1]], model.u_nohalo,-model.u_nohalo[:,[end, end-1]],dims=2),zeros(P_temp.T,halo,nuy+2*halo),dims=1))
    # model.S.Prog.v .= scale*(cat(zeros(P_temp.T,nvx+2*halo,halo),cat(-model.v_nohalo[[2; 1],:],model.v_nohalo,-model.v_nohalo[[end; end-1],:],dims=1),zeros(P_temp.T,nvx+2*halo,halo),dims=2))

    return exp2_integrate(model)

end

function NLPModels.grad!(model, param_guess, G)

    P_temp = ShallowWaters.Parameter(T=model.S.parameters.T;
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
        Ndays=model.S.parameters.Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_spinup_noforcing/"
    )

    model.S = ShallowWaters.model_setup(P_temp)
    model.J = 0.0
    model.t = 0.0
    model.j = 1

    snaps = Int(floor(sqrt(model.S.grid.nt)))
    revolve = Revolve(
        snaps;
        verbose=0,
        gc=true,
        write_checkpoints=false
    )

    current = 1
    for m in (model.S.Prog.u, model.S.Prog.v)#, model.S.Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    # modifying initial condition with the halo
    # model.u_nohalo .= scale_inv*model.S.Prog.u[halo+1:end-halo,halo+1:end-halo]
    # model.v_nohalo .= scale_inv*model.S.Prog.v[halo+1:end-halo,halo+1:end-halo]
    # current = 1
    # for m in (model.u_nohalo, model.v_nohalo)#, model.S.Prog.η)
    #     sz = prod(size(m))
    #     m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
    #     current += sz
    # end

    # add halo back for integrating
    # model.S.Prog.u .= scale*(cat(zeros(P_temp.T,halo,nuy+2*halo),cat(-model.u_nohalo[:,[2,1]], model.u_nohalo,-model.u_nohalo[:,[end, end-1]],dims=2),zeros(P_temp.T,halo,nuy+2*halo),dims=1))
    # model.S.Prog.v .= scale*(cat(zeros(P_temp.T,nvx+2*halo,halo),cat(-model.v_nohalo[[2; 1],:],model.v_nohalo,-model.v_nohalo[[end; end-1],:],dims=1),zeros(P_temp.T,nvx+2*halo,halo),dims=2))

    dmodel = Enzyme.make_zero(model)

    J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        exp2_cpintegrate,
        Active,
        Duplicated(model, dmodel),
        Const(revolve)
    )[2]


    G .= [vec(dmodel.S.Prog.u); vec(dmodel.S.Prog.v); vec(dmodel.S.Prog.η)]

    return nothing

end

function compute_initcond_newoptimizer()

    Ndays = 20
    N = 20
    sigma_data = 0.1                # this and the init cond are what the result from optimization used, be careful in adjusting
    sigma_initcond = 0.1
    data_steps = 225:224:Ndays*225

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]

    # setup all models
    adj_model, ekf_model, param_guess, S_pred, pred_states, P_pred, true_states = exp2_model_setup(Float64,
        Ndays,
        N,
        sigma_data,
        sigma_initcond,
        data_steps,
        data_spots
    )

    # run the ensemble Kalman filter
    ekf_avgu, ekf_avgv = run_ensemble_kf(ekf_model, param_guess)

    # run the adjoint optimization
    qn_options = MadNLP.QuasiNewtonOptions(; max_history=50)
    result = madnlp(
        adj_model;
        # linear_solver=LapackCPUSolver,
        hessian_approximation=MadNLP.CompactLBFGS,
        quasi_newton_options=qn_options,
        max_iter=100
    )

    # integrate with the result from the optimization
    S_adj = ShallowWaters.model_setup(P_pred)
    current = 1
    for m in (S_adj.Prog.u, S_adj.Prog.v, S_adj.Prog.η)
        sz = prod(size(m))
        m .= reshape(result.solution[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    _, states_adj = exp2_generate_data(S_adj, data_steps, data_spots, sigma_data)

    return ekf_avgu, ekf_avgv, G, dS, data, true_states, result, S_adj, states_adj, S_pred, pred_states

end

function ignore(result)

    Ndays = 10
    sigma_initcond = 0.1

    P_true = ShallowWaters.Parameter(T=Float64,
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        tracer_advection=false,
        tracer_relaxation=false,
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./128_postspinup_1year_noslipbc_epsetup/",
        init_starti=1
    )

    S_true = ShallowWaters.model_setup(P_true)

    halo = S_true.grid.halo
    scale_inv = S_true.constants.scale_inv
    scale = S_true.constants.scale
    nuy = S_true.grid.nuy
    nvx = S_true.grid.nvx

    P_temp = ShallowWaters.Parameter(T=Float64,
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        tracer_advection=false,
        tracer_relaxation=false,
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./128_postspinup_1year_noslipbc_epsetup/",
        init_starti=1
    )

    S_temp = ShallowWaters.model_setup(P_temp)

    data_steps = 225:224:Ndays*225

    udata = ncread("./128_postspinup_1year_noslipbc_epsetup/u.nc", "u")
    vdata = ncread("./128_postspinup_1year_noslipbc_epsetup/v.nc", "v")
    etadata = ncread("./128_postspinup_1year_noslipbc_epsetup/eta.nc", "eta")

    data = zeros(128*127*2, Ndays)# + 128*128, Ndays)
    for j = 2:Ndays+1
        data[:,j-1] .= [vec(udata[:,:,j]); vec(vdata[:,:,j])]#; vec(etadata[:,:,j])]
    end

    Prog_pred = ShallowWaters.PrognosticVars{Float32}(ShallowWaters.remove_halo(S_temp.Prog.u,
        S_temp.Prog.v,
        S_temp.Prog.η,
        S_temp.Prog.sst,
        S_temp)...
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

    # for n = 1:5
    #     for m = 1:5
    #         etarand = randn(4)
    #         for k = 1:128
    #             for j = 1:128
    #                 etapert[k,j] += sigma_initcond * etarand[1] * cos((pi * n / 128) * k)*cos(pi * m / 128 * j)
    #                     + sigma_initcond * etarand[2] * cos((pi * n / 128) * k)*sin(pi * m / 128 * j)
    #                     + sigma_initcond * etarand[3] * sin((pi * n / 128) * k)*cos(pi * m / 128 * j)
    #                     + sigma_initcond * etarand[4] * sin((pi * n / 128) * k)*sin(pi * m / 128 * j)
    #             end
    #         end
    #     end
    # end

    Prog_pred.u = Prog_pred.u + upert
    Prog_pred.v = Prog_pred.v + vpert
    Prog_pred.η = Prog_pred.η + etapert

    S_pred = ShallowWaters.model_setup(P_temp)
    uic,vic,etaic = ShallowWaters.add_halo(Prog_pred.u,Prog_pred.v,Prog_pred.η,Prog_pred.sst,S_pred)

    S_pred.Prog.u = uic
    S_pred.Prog.v = vic
    S_pred.Prog.η = etaic

    P_temp = ShallowWaters.Parameter(T=Float64;output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        tracer_advection=false,
        tracer_relaxation=false,
        α=2,
        nx=128,
        Ndays=5,
        initial_cond="rest"
    )

    S_nlp = ShallowWaters.model_setup(P_temp)

    # modifying initial condition with the halo
    u_nohalo = scale_inv*S_nlp.Prog.u[halo+1:end-halo,halo+1:end-halo]
    v_nohalo = scale_inv*S_nlp.Prog.v[halo+1:end-halo,halo+1:end-halo]
    current = 1
    for m in (u_nohalo, v_nohalo)#, model.S.Prog.η)
        sz = prod(size(m))
        m .= reshape(result.solution[current:(current + sz - 1)], size(m)...)
        current += sz
    end

    # add halo back for integrating
    S_nlp.Prog.u .= scale*(cat(zeros(P_temp.T,halo,nuy+2*halo),cat(-u_nohalo[:,[2,1]], u_nohalo,-u_nohalo[:,[end, end-1]],dims=2),zeros(P_temp.T,halo,nuy+2*halo),dims=1))
    S_nlp.Prog.v .= scale*(cat(zeros(P_temp.T,nvx+2*halo,halo),cat(-v_nohalo[[2; 1],:],v_nohalo,-v_nohalo[[end; end-1],:],dims=1),zeros(P_temp.T,nvx+2*halo,halo),dims=2))


    # current = 1
    # for m in (S_nlp.Prog.u, S_nlp.Prog.v)#, S_nlp.Prog.η)
    #     sz = prod(size(m))
    #     m .= reshape(results.solution[current:(current + sz - 1)], size(m)...)
    #     current += sz
    # end

    Prog_nlp = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(S_nlp.Prog.u,
        S_nlp.Prog.v,
        S_nlp.Prog.η,
        S_nlp.Prog.sst,
        S_nlp)...
    )

    println("difference ", norm(Prog_nlp.u - u_nohalo))

    Prog_true = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(S_true.Prog.u,
        S_true.Prog.v,
        S_true.Prog.η,
        S_true.Prog.sst,
        S_true)...
    )

    Prog_true10 = ShallowWaters.time_integration(S_true)
    Prog_pred10 = ShallowWaters.time_integration(S_pred)
    Prog_nlp10 = ShallowWaters.time_integration(S_nlp)

    return Prog_true, Prog_pred, Prog_nlp, upert, vpert, etapert, Prog_true10, Prog_pred10, Prog_nlp10


end

function finite_difference(Ndays, xcoord, ycoord)

    # Type precision
    T = Float64

    P_temp = ShallowWaters.Parameter(T=T;
        output=false,
        L_ratio=1,
        g=9.81,
        H=500,
        wind_forcing_x="double_gyre",
        Lx=3840e3,
        seasonal_wind_x=false,
        topography="flat",
        bc="nonperiodic",
        bottom_drag="quadratic",
        tracer_advection=false,
        tracer_relaxation=false,
        α=2,
        nx=128,
        Ndays=Ndays,
        initial_cond="ncfile",
        initpath="./128_postspinup_1year_noslipbc_epsetup/",
        init_starti=1
    )

    S0 = ShallowWaters.model_setup(P_temp)

    snaps = Int(floor(sqrt(S0.grid.nt)))
    revolve = Revolve(
        snaps;
        verbose=0,
        gc=true,
        write_checkpoints=false,
        write_checkpoints_filename = "",
        write_checkpoints_period = 2274
    )

    data_steps = 225:224:Ndays*225

    udata = ncread("./128_postspinup_1year_noslipbc_epsetup/u.nc", "u")
    vdata = ncread("./128_postspinup_1year_noslipbc_epsetup/v.nc", "v")
    etadata = ncread("./128_postspinup_1year_noslipbc_epsetup/eta.nc", "eta")

    data = zeros(128*127*2 + 128*128, Ndays)
    for j = 2:Ndays+1
        data[:,j-1] .= [vec(udata[:,:,j]); vec(vdata[:,:,j]); vec(etadata[:,:,j])]
    end

    param_guess = [vec(S0.Prog.u);vec(S0.Prog.v);vec(S0.Prog.η)]

    meta = NLPModelMeta(length([vec(S0.Prog.u); vec(S0.Prog.v); vec(S0.Prog.η)]);ncon=0,nnzh=0,x0=param_guess)

    S1 = deepcopy(S0)
    chkp1 = InitCondModel{T, typeof(param_guess)}(meta,
        Counters(),
        S1,
        0.0,
        data,
        data_steps,
        1,
        1,
        0.0
    )
    dchkp1 = Enzyme.make_zero(chkp1)

    # Enzyme deriv
    J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        cpintegrate,
        Active,
        Duplicated(chkp1, dchkp1),
        Const(revolve)
    )[2]
    enzyme_deriv = dchkp1.S.Prog.u[xcoord, ycoord]
    println("Loss when using Enzyme + Checkpointing: $J")
    println("Enzyme derivative: $enzyme_deriv")

    S2 = deepcopy(S0)
    chkp2 = InitCondModel{T, typeof(param_guess)}(meta,
        Counters(),
        S2,
        0.0,
        data,
        data_steps,
        1,
        1,
        0.0
    )

    @time unperturbed_loss = cpintegrate(chkp2, revolve)
    println("Loss when not using Enzyme: $unperturbed_loss")

    steps = [100, 50, 30, 20, 10, 1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    diffs = []
    for s in steps

        S3 = deepcopy(S0)
        chkp3 = InitCondModel{T, typeof(param_guess)}(meta,
            Counters(),
            S3,
            0.0,
            data,
            data_steps,
            1,
            1,
            0.0
        )

        chkp3.S.Prog.u[xcoord,ycoord] += s

        J = cpintegrate(chkp3, revolve)
        push!(diffs, (J - unperturbed_loss) / s)

    end

    println("Finite difference result: $diffs")

end