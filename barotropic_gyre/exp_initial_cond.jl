"""
Loss is the difference of u, v, and η with data for same fields
Aim to tune the initial condition, all of u, v, and η are perturbed
Can play with how sparse the data is both temporally and spatially
"""

mutable struct exp_initcond_adjmodel{T, S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    S::ShallowWaters.ModelSetup{T,T}        # model structure
    u_nohalo::Array{T,2}
    v_nohalo::Array{T,2}
    J::Float64                              # objective value
    data::Array{T, 2}
    data_steps::StepRange{Int, Int}         # when data is assimilated temporally
    data_spots::Array{Int,1}                # where data is located spatially, grid coordinates
    j::Int                                  # for keeping track of location in data
    i::Int                                  # timestep iterator, needed for checkpointing
    t::Int64                                # model time (e.g. dt * i)
end

mutable struct exp_initcond_ekfmodel{T}
    S::ShallowWaters.ModelSetup{T,T}        # model struct for adjoint
    N::Int                                  # number of ensemble members
    data::Array{T, 2}                       # data to be assimilated
    sigma_initcond::T
    sigma_data::T
    data_steps::StepRange{Int, Int}         # when data is assimilated
    data_spots::Array{Int, 1}               # where data is located, grid coordinates
    j::Int                                  # for keeping track of location in data
    t::Int64                                # model timestep (i * Δt)
end

function initcond_model_setup(T, Ndays, N, sigma_data, sigma_initcond, data_steps, data_spots)

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
        initpath="./data_files_forkf/128_postspinup_1year_noslipbc_epsetup",
        init_starti=1
    )

    S_true = ShallowWaters.model_setup(P_pred)

    println("norm of true u ", norm(S_true.Prog.u))
    println("norm of true v ", norm(S_true.Prog.v))
    println("norm of true eta ", norm(S_true.Prog.η))

    udata = ncread("./data_files_forkf/128_postspinup_1year_noslipbc_epsetup/u.nc", "u")
    vdata = ncread("./data_files_forkf/128_postspinup_1year_noslipbc_epsetup/v.nc", "v")
    etadata = ncread("./data_files_forkf/128_postspinup_1year_noslipbc_epsetup/eta.nc", "eta")

    data = zeros(128*127*2 + 128^2, Ndays)
    # this is offset by one because the initial condition is included in the above saved states
    for j = 2:Ndays+1
        perturbed_udata = udata[:,:,j] .+ sigma_data .* randn(size(udata[:,:,1]))
        perturbed_vdata = vdata[:,:,j] .+ sigma_data .* randn(size(vdata[:,:,1]))
        perturbed_etadata = etadata[:,:,j] .+ sigma_data .* randn(size(etadata[:,:,1]))
        data[:,j-1] .= [vec(perturbed_udata); vec(perturbed_vdata); vec(perturbed_etadata)]
    end

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

    println("norm of predicted - true u ", norm(S_pred.Prog.u - S_true.Prog.u))
    println("norm of predicted - true v ", norm(S_pred.Prog.v - S_true.Prog.v))
    println("norm of predicted - true eta ", norm(S_pred.Prog.η - S_true.Prog.η))

    param_guess = [vec(Prog_pred.u); vec(Prog_pred.v); vec(Prog_pred.η)]

    Spred = deepcopy(S_pred)
    pred_states = hourly_save_run(Spred)

    # doing deepcopies to be 100% sure that I'm not messing with the model at any point during setup
    Skf = deepcopy(S_pred)
    Sadj = deepcopy(S_pred)

    meta = NLPModelMeta(length(param_guess); ncon=0, nnzh=0,x0=param_guess)
    adj_model = exp_initcond_adjmodel{T, typeof(param_guess)}(meta,
        Counters(),
        Sadj,
        zeros(size(Prog_pred.u)),
        zeros(size(Prog_pred.v)),
        0.0,
        data,
        data_steps,
        data_spots,
        1,
        1,
        0.0
    )

    ekf_model = exp_initcond_ekfmodel{T}(
        Skf,
        N,
        data,
        sigma_initcond,
        sigma_data,
        data_steps,
        data_spots,
        1,
        0
    )

    return adj_model, ekf_model, param_guess, S_pred, pred_states, P_pred
end

function initcond_cpintegrate(chkp, scheme)::Float64

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

        t = chkp.t
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

        tempuv = [vec(temp.u); vec(temp.v); vec(temp.η)]
        chkp.J += sum((tempuv[chkp.data_spots] - chkp.data[:, chkp.j][chkp.data_spots]).^2)

        chkp.j += 1
    end

    copyto!(chkp.S.Prog.u, chkp.S.Diag.RungeKutta.u0)
    copyto!(chkp.S.Prog.v, chkp.S.Diag.RungeKutta.v0)
    copyto!(chkp.S.Prog.η, chkp.S.Diag.RungeKutta.η0)
    end

    return chkp.J

end

function initcond_integrate(chkp)::Float64

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

        t = chkp.t
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

            tempuv = [vec(temp.u); vec(temp.v); vec(temp.η)]
            chkp.J += sum((tempuv[chkp.data_spots] - chkp.data[:, chkp.j][chkp.data_spots]).^2)
            chkp.j += 1

        end

        copyto!(chkp.S.Prog.u, chkp.S.Diag.RungeKutta.u0)
        copyto!(chkp.S.Prog.v, chkp.S.Diag.RungeKutta.v0)
        copyto!(chkp.S.Prog.η, chkp.S.Diag.RungeKutta.η0)
    end

    return chkp.J

end

function NLPModels.obj(model, param_guess)

    P_temp = ShallowWaters.Parameter(T=model.S.parameters.T;
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
        Ndays=model.S.parameters.Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_postspinup_1year_noslipbc_epsetup",
        init_starti=1
    )

    # ensuring these fields are reset between iterations of the optimizer
    model.S = ShallowWaters.model_setup(P_temp)
    model.J = 0.0
    model.t = 0.0
    model.j = 1

    Prog = ShallowWaters.PrognosticVars{model.S.parameters.T}(ShallowWaters.remove_halo(model.S.Prog.u,
        model.S.Prog.v,
        model.S.Prog.η,
        model.S.Prog.sst,
        model.S)...
    )
    current = 1
    for m in (Prog.u, Prog.v, Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    umodified,vmodified,eta_modified = ShallowWaters.add_halo(Prog.u,Prog.v,Prog.η,Prog.sst,model.S)

    model.S.Prog.u .= umodified
    model.S.Prog.v .= vmodified
    model.S.Prog.η .= eta_modified

    return initcond_integrate(model)

end

function NLPModels.grad!(model, param_guess, G)

    P_temp = ShallowWaters.Parameter(T=model.S.parameters.T;
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
        Ndays=model.S.parameters.Ndays,
        initial_cond="ncfile",
        initpath="./data_files_forkf/128_postspinup_1year_noslipbc_epsetup",
        init_starti=1
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

    # remove halo
    Prog = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(model.S.Prog.u,
        model.S.Prog.v,
        model.S.Prog.η,
        model.S.Prog.sst,
        model.S)...
    )
    # place the current guess as the initial conditions
    current = 1
    for m in (Prog.u, Prog.v, Prog.η)
        sz = prod(size(m))
        m .= reshape(param_guess[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    umodified,vmodified,etamodified = ShallowWaters.add_halo(Prog.u,Prog.v,Prog.η,Prog.sst,model.S)

    model.S.Prog.u .= umodified
    model.S.Prog.v .= vmodified
    model.S.Prog.η .= etamodified

    dmodel = Enzyme.make_zero(model)

    J = autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        initcond_cpintegrate,
        Active,
        Duplicated(model, dmodel),
        Const(revolve)
    )[2]

    dProg = ShallowWaters.PrognosticVars{Float64}(ShallowWaters.remove_halo(dmodel.S.Prog.u,
        dmodel.S.Prog.v,
        dmodel.S.Prog.η,
        dmodel.S.Prog.sst,
        model.S)...
    )

    G .= [vec(dProg.u); vec(dProg.v); vec(dProg.η)]

    return nothing

end

function run_initcond(Ndays, sigma_data, sigma_initcond)

    # number of ensemble members, typically leaving this 20
    N = 10

    # daily data
    data_steps = 225:224:Ndays*225

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    # we want data for all of u, v, and η for this experiment
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)
    data_spotseta = vec((Xu.-1) .* 128 + Yu) .+ (128*127*2)
    data_spots = [data_spotsu; data_spotsv; data_spotseta]

    # setup all models
    adj_model, ekf_model, param_guess, S_pred, states_pred, P_pred = initcond_model_setup(Float64,
        Ndays,
        N,
        sigma_data,
        sigma_initcond,
        data_steps,
        data_spots
    )

    # run the ensemble Kalman filter
    ekf_avgu, ekf_avgv, ekf_avgeta = run_ensemble_kf(ekf_model, param_guess;udata=true,vdata=true,etadata=true)

    # run the adjoint optimization
    qn_options = MadNLP.QuasiNewtonOptions(; max_history=100)
    result = madnlp(
        adj_model;
        # linear_solver=LapackCPUSolver,
        hessian_approximation=MadNLP.CompactLBFGS,
        quasi_newton_options=qn_options,
        max_iter=200
    )

    # integrate with the result from the optimization
    S_adj = ShallowWaters.model_setup(P_pred)
    uic = S_adj.parameters.T.(zeros(127,128))
    vic = S_adj.parameters.T.(zeros(128,127))
    etaic = S_adj.parameters.T.(zeros(128,128))
    current = 1
    for m in (uic, vic, etaic)
        sz = prod(size(m))
        m .= reshape(result.solution[current:(current + sz - 1)], size(m)...)
        current += sz
    end
    utuned,vtuned,etatuned,_ = ShallowWaters.add_halo(uic,vic,etaic,zeros(128,128),S_adj)
    S_adj.Prog.u = utuned
    S_adj.Prog.v = vtuned
    S_adj.Prog.η = etatuned
    states_adj = hourly_save_run(S_adj)

    return ekf_avgu, ekf_avgv, ekf_avgeta, result, S_adj, states_adj, S_pred, states_pred

end

function initcond_plots()

    # ekf_avgu, ekf_avgv, true_states, result, S_adj, states_adj, S_pred, pred_states

    # number of days to run the integration
    Ndays = 30

    # number of ensemble members, typically leaving this 20
    N = 10

    # trying with extra noisy data
    sigma_data = 0.0001
    sigma_initcond = 0.0001
    data_steps = 225:224:Ndays*225

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    # we want data for all of u, v, and η for this experiment
    data_spotsu = vec((Xu.-1) .* 127 + Yu);
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127);
    data_spotseta = vec((Xu.-1) .* 128 + Yu) .+ (128*127*2);
    data_spots = [data_spotsu; data_spotsv; data_spotseta];

    udata = ncread("./data_files_forkf/128_postspinup_1year_noslipbc_epsetup/u.nc", "u");
    vdata = ncread("./data_files_forkf/128_postspinup_1year_noslipbc_epsetup/v.nc", "v");
    etadata = ncread("./data_files_forkf/128_postspinup_1year_noslipbc_epsetup/eta.nc", "eta");

    ekf_avgu = load_object("./initcond_dailydata_allofuveta/ekf_avgu_initcond.jld2");
    ekf_avgv = load_object("./initcond_dailydata_allofuveta/ekf_avgv_initcond.jld2");
    states_adj = load_object("./initcond_dailydata_allofuveta/states_adj_initcond.jld2");
    states_pred = load_object("./initcond_dailydata_allofuveta/pred_states_initcond.jld2");

    # location of data spatially on true fields
    fig = Figure(size=(1000, 500));
    ax1, hm1 = heatmap(fig[1,1], true_states[end].u,
        colormap=:balance,
        colorrange=(-maximum(true_states[end].u),
        maximum(true_states[end].u)),
        axis=(xlabel=L"x", ylabel=L"y", title=L"u(t = 30 \; \text{days}, x, y)"),
    );
    scatter!(ax1, vec(Xu), vec(Yu), color=:green);
    Colorbar(fig[1,2], hm1)
    ax2, hm2 = heatmap(fig[1,3], true_states[end].v,
        colormap=:balance,
        colorrange=(-maximum(true_states[end].v),
        maximum(true_states[end].v)),
        axis=(xlabel=L"x", ylabel=L"y", title=L"v(t = 30 \; \text{days}, x, y)"),
    );
    scatter!(ax2, vec(Xu), vec(Yu), color=:green);
    Colorbar(fig[1,4], hm2);

    # predicted states
    fig = Figure(size=(900,400), fontsize=15);
    t = 748
    ax1, hm1 = heatmap(fig[1,1],
        states_pred[t].u,
        colormap=:balance,
        colorrange=(-maximum(abs.(udata[:,:,30])), maximum(abs.(udata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{u}(t = 30 \; \text{days}, x, y, -)")
    )
    Colorbar(fig[1,2], hm1)

    ax2, hm2 = heatmap(fig[1,3], 
        states_pred[t].v,
        colormap=:balance,
        colorrange=(-maximum(abs.(udata[:,:,30])), maximum(abs.(udata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y, -)")
    );
    Colorbar(fig[1,4], hm2)

    # results for u and v fields
    fig = Figure(size=(1000,600));
    t = 748

    ax1, hm1 = heatmap(fig[1,1],
        # udata[:,:,30],
        states_pred[t].u,
        colormap=:balance,
        colorrange=(-maximum(abs.(udata[:,:,30])), maximum(abs.(udata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{u}(t = 30 \; \text{days}, x, y, -)")
    )
    Colorbar(fig[1,2], hm1)

    ax2, hm2 = heatmap(fig[1,3], 
        states_adj[t].u,
        colormap=:balance,
        colorrange=(-maximum(abs.(udata[:,:,30])), maximum(abs.(udata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{u}(t = 30 \; \text{days}, x, y, +)")
    );
    Colorbar(fig[1,4], hm2)

    ax3, hm3 = heatmap(fig[1, 5], 
        ekf_avgu[t],
        colormap=:balance,
        colorrange=(-maximum(abs.(udata[:,:,30])), maximum(abs.(udata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{u}(t = 30 \; \text{days}, x, y)")
    );
    Colorbar(fig[1,6], hm3)

    ax4, hm4 = heatmap(fig[2,1],
        # vdata[:,:,30],
        states_pred[t].v,
        colormap=:balance,
        colorrange=(-maximum(abs.(vdata[:,:,30])), maximum(abs.(vdata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y, -)")
    )
    Colorbar(fig[2,2], hm4)

    ax5, hm5 = heatmap(fig[2,3], 
        states_adj[t].v,
        colormap=:balance,
        colorrange=(-maximum(abs.(vdata[:,:,30])), maximum(abs.(vdata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y, +)")
    );
    Colorbar(fig[2,4], hm5)

    ax6, hm6 = heatmap(fig[2, 5], 
        ekf_avgv[t],
        colormap=:balance,
        colorrange=(-maximum(abs.(vdata[:,:,30])), maximum(abs.(vdata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y)")
    );
    Colorbar(fig[2,6], hm6)

    # results for v field
    fig = Figure(size=(1000,300));
    t = 748

    ax0, hm0 = heatmap(fig[1,1],
        vdata[:,:,30],
        colormap=:balance,
        colorrange=(-maximum(abs.(vdata[:,:,30])), maximum(abs.(vdata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"v(t = 30 \; \text{days}, x, y)")
    )
    Colorbar(fig[1,2], hm0)

    ax1, hm1 = heatmap(fig[1,3], 
        states_adj[t].v,
        colormap=:balance,
        colorrange=(-maximum(abs.(vdata[:,:,30])), maximum(abs.(vdata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y, +)")
    );
    Colorbar(fig[1,4], hm1)

    ax2, hm2 = heatmap(fig[1, 5], 
        ekf_avgv[t],
        colormap=:balance,
        colorrange=(-maximum(abs.(vdata[:,:,30])), maximum(abs.(vdata[:,:,30]))),
        axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{v}(t = 30 \; \text{days}, x, y)")
    );
    Colorbar(fig[1,6], hm2)

    ax2, hm2 = heatmap(fig[1,3],
        abs.(vdata[:,:,30] .- states_adj[t].v),
        colormap=:amp,
        # colorrange=(0,
        # 1.5),
        axis=(xlabel=L"x", ylabel=L"y", title=L"|v(x,y) - \tilde{v}(x, y, +)|")
    )
    Colorbar(fig[1,4], hm2)


    ax4, hm4 = heatmap(fig[2,3], abs.(vdata[:,:,30] .- ekf_avgv[t]),
        colormap=:amp,
        # colorrange=(0,
        # 1.5),
        axis=(xlabel=L"x", ylabel=L"y", title=L"|v(x,y) - \tilde{v}(x, y)|")
    )
    Colorbar(fig[2,4], hm4)

    # energy plots

    fig = Figure(size=(600, 500));
    ax1, hm1 = heatmap(fig[1,1], (true_states[end].u[:, 1:end-1].^2 .+ true_states[end].v[1:end-1, :].^2),
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\mathcal{E}"),
    colorrange=(0,
    maximum(true_states[end].u[:, 1:end-1].^2 .+ true_states[end].v[1:end-1, :].^2))
    );
    Colorbar(fig[1,2], hm1)

    fig = Figure(size=(800, 700));
    t = 748

    ax1, hm1 = heatmap(fig[1,1], pred_states[t].u[:, 1:end-1].^2 .+ pred_states[t].v[1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}(+)"),
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    );
    Colorbar(fig[1,2], hm1)

    ax2, hm2 = heatmap(fig[1,3], abs.((true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2) .- (states_adj[t].u[:, 1:end-1].^2 .+ states_adj[t].v[1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}(+)|")
    )
    Colorbar(fig[1,4], hm2)

    ax3, hm3 = heatmap(fig[2, 1], ekf_avgu[t][:, 1:end-1].^2 .+ ekf_avgv[t][1:end-1, :].^2,
    colormap=:amp,
    axis=(xlabel=L"x", ylabel=L"y", title=L"\tilde{\mathcal{E}}"),
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    );
    Colorbar(fig[2,2], hm3)

    ax4, hm4 = heatmap(fig[2,3], abs.((true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2) .- (ekf_avgu[t][:, 1:end-1].^2 .+ ekf_avgv[t][1:end-1, :].^2)),
    colormap=:amp,
    colorrange=(0,
    maximum(true_states[t].u[:, 1:end-1].^2 .+ true_states[t].v[1:end-1, :].^2)),
    axis=(xlabel=L"x", ylabel=L"y", title=L"|\mathcal{E} - \tilde{\mathcal{E}}|")
    )
    Colorbar(fig[2,4], hm4)

    # spatially averaged energy

    true_energy = zeros(673)
    pred_energy = zeros(673)
    ekf_energy = zeros(673)

    for t = 1:673

        true_energy[t] = (sum(true_states[t].u.^2) + sum(true_states[t].v.^2)) / (128 * 127)
        pred_energy[t] = (sum(pred_states[t].u.^2) + sum(pred_states[t].v.^2)) / (128 * 127)
        ekf_energy[t] = (sum(ekf_avgu[t].^2) + sum(ekf_avgv[t].^2)) / (128 * 127)

    end

    fig = Figure(size=(800, 700));
    lines(fig[1,1], true_energy, label="Truth")
    lines!(fig[1,1], pred_energy, label="Pred")
    lines!(fig[1,1], ekf_energy, label="EKF")
    axislegend()

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

    for t = 1:498
        # up_adj[:,t] = power(periodogram(states_adj[t].u; radialavg=true))
        # vp_adj[:,t] = power(periodogram(states_adj[t].v; radialavg=true))

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
    # lines(fig2[1,1], adj_wl[2:end], up_adj[2:end,t] + vp_adj[2:end,t], label="Adjoint", axis=(xscale=log10,yscale=log10,xlabel="Wavelength (km)", ylabel="KE(k)", xreversed=true, xticks=[100, 30, 10, 2]))
    lines(fig2[1,1], ekf_wl[2:end], up_ekf[2:end,t] + vp_ekf[2:end,t], label="EKF")
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

# not sure that this has been updated, can fix
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