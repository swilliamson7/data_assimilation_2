
module Checkpointing
using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules
import Enzyme.EnzymeCore
using LinearAlgebra

function checkpoint_struct_for(body::Function, scheme, model, range)
    for gensym() in range
        body(model)
    end
    return model
end

struct Revolve{MT}
end

macro checkpoint_struct(alg, model, loop)
    if loop.head == :for
        body = loop.args[2]
        iterator = loop.args[1].args[1]
        from = loop.args[1].args[2].args[2]
        to = loop.args[1].args[2].args[3]
        range = loop.args[1].args[2]
        ex = quote
            let
                if !isa($range, UnitRange{Int64})
                    error("Checkpointing.jl: Only UnitRange{Int64} is supported.")
                end
                $iterator = $from
                $model = Checkpointing.checkpoint_struct_for(
                    $alg,
                    $model,
                    $(loop.args[1].args[2]),
                ) do $model
                    $body
                    $iterator += 1
                    nothing
                end
            end
        end
    elseif loop.head == :while
        ex = quote
            function condition($model)
                $(loop.args[1])
            end
            $model =
                Checkpointing.checkpoint_struct_while($alg, $model, condition) do $model
                    $(loop.args[2])
                    nothing
                end
        end
    else
        error("Checkpointing.jl: Unknown loop construct.")
    end
    esc(ex)
end


function augmented_primal(
    config,
    func::Const{typeof(Checkpointing.checkpoint_struct_for)},
    ret,
    body,
    alg,
    model,
    range,
)
    tape_model = deepcopy(model.val)
    func.val(body.val, alg.val, model.val, range.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_model,))
    else
        return AugmentedReturn(nothing, nothing, (tape_model,))
    end
end
function rev_checkpoint_struct_for(
    config,
    body::Function,
    alg::Revolve,
    model_input::MT,
    shadowmodel::MT,
    range,
) where {MT}
    # chkp = model_input
    # model = model_input
    # dchkp = model.dval
    # println("J2 : $(chkp.S.parameters.J)")
    # println("dJ : $(dchkp.S.parameters.J)")
    # println("i : $(chkp.S.parameters.i)")
    # println("j : $(chkp.S.parameters.j)")
    dup = Duplicated(model_input, shadowmodel)
    Enzyme.autodiff(
        EnzymeCore.set_runtime_activity(Reverse, config),
        Const(body), dup,
    )
    # dS = shadowmodel.S
    # G = [vec(dS.Prog.u); vec(dS.Prog.v); vec(dS.Prog.η)]
    # println("Norm($step) autodiff: ", norm(G))
    # println("reverse G: ", norm(shadowmodel.S.Prog.u))
end

function reverse(
    config,
    ::Const{typeof(Checkpointing.checkpoint_struct_for)},
    dret::Type{<:Const},
    tape,
    body,
    alg,
    model::Duplicated,
    range,
)
    (model_input,) = tape
    rev_checkpoint_struct_for(
        config,
        body.val,
        alg.val,
        model_input,
        model.dval,
        range.val,
    )
    return (nothing, nothing, nothing, nothing)
end
export augmented_primal, reverse, Revolve, @checkpoint_struct
end

using .Checkpointing
include("BarotropicGyre.jl")

mutable struct MyParameters
    J::Float64
    i::Int
end

mutable struct MyPrognosticVars
    u::Array{Float32, 2}
end

mutable struct MyModelSetup
    parameters::MyParameters
    Prog::MyPrognosticVars
end

mutable struct Chkp
# mutable struct Chkp{MT}
    # S::MT
    S::ShallowWaters.ModelSetup
    data::Matrix{Float32}
    data_spots::Vector{Int}
end

function generate_data(S_true, data_spots, sigma_data; compute_freq=false)

    data = Float32.(zeros(length(data_spots), length(S_true.parameters.data_steps)))
    true_states = []

    # setup that happens prior to the actual integration
    Diag = S_true.Diag
    Prog = S_true.Prog

    @unpack u,v,η,sst = Prog
    @unpack u0,v0,η0 = Diag.RungeKutta
    @unpack u1,v1,η1 = Diag.RungeKutta
    @unpack du,dv,dη = Diag.Tendencies
    @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    @unpack um,vm = Diag.SemiLagrange

    @unpack dynamics,RKo,RKs,tracer_advection = S_true.parameters
    @unpack time_scheme,compensated = S_true.parameters
    @unpack RKaΔt,RKbΔt = S_true.constants
    @unpack Δt_Δ,Δt_Δs = S_true.constants

    @unpack nt,dtint = S_true.grid
    @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S_true.grid

    freqpoweru = []
    freqpowerv = []

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(Diag.VolumeFluxes.h,η,S_true.forcing.H)
    ShallowWaters.Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    ShallowWaters.advection_coriolis!(urhs,vrhs,ηrhs,Diag,S_true)
    ShallowWaters.PVadvection!(Diag,S_true)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)
    j = 1

    for t = 1:S_true.grid.nt

        Diag = S_true.Diag
        Prog = S_true.Prog

        @unpack u,v,η,sst = Prog
        @unpack u0,v0,η0 = Diag.RungeKutta
        @unpack u1,v1,η1 = Diag.RungeKutta
        @unpack du,dv,dη = Diag.Tendencies
        @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
        @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

        @unpack um,vm = Diag.SemiLagrange

        @unpack dynamics,RKo,RKs,tracer_advection = S_true.parameters
        @unpack time_scheme,compensated = S_true.parameters
        @unpack RKaΔt,RKbΔt = S_true.constants
        @unpack Δt_Δ,Δt_Δs = S_true.constants

        @unpack nt,dtint = S_true.grid
        @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S_true.grid
        i = S_true.parameters.i

        # ghost point copy for boundary conditions
        ShallowWaters.ghost_points!(u,v,η,S_true)
        copyto!(u1,u)
        copyto!(v1,v)
        copyto!(η1,η)

        if compensated
            fill!(du_sum,zero(Tprog))
            fill!(dv_sum,zero(Tprog))
            fill!(dη_sum,zero(Tprog))
        end

        for rki = 1:RKo
            if rki > 1
                ShallowWaters.ghost_points!(u1,v1,η1,S_true)
            end

            # type conversion for mixed precision
            u1rhs = convert(Diag.PrognosticVarsRHS.u,u1)
            v1rhs = convert(Diag.PrognosticVarsRHS.v,v1)
            η1rhs = convert(Diag.PrognosticVarsRHS.η,η1)

            ShallowWaters.rhs!(u1rhs,v1rhs,η1rhs,Diag,S_true,t)          # momentum only
            ShallowWaters.continuity!(u1rhs,v1rhs,η1rhs,Diag,S_true,t)   # continuity equation

            if rki < RKo
                ShallowWaters.caxb!(u1,u,RKbΔt[rki],du)   #u1 .= u .+ RKb[rki]*Δt*du
                ShallowWaters.caxb!(v1,v,RKbΔt[rki],dv)   #v1 .= v .+ RKb[rki]*Δt*dv
                ShallowWaters.caxb!(η1,η,RKbΔt[rki],dη)   #η1 .= η .+ RKb[rki]*Δt*dη
            end

            if compensated      # accumulate tendencies
                ShallowWaters.axb!(du_sum,RKaΔt[rki],du)
                ShallowWaters.axb!(dv_sum,RKaΔt[rki],dv)
                ShallowWaters.axb!(dη_sum,RKaΔt[rki],dη)
            else    # sum RK-substeps on the go
                ShallowWaters.axb!(u0,RKaΔt[rki],du)          #u0 .+= RKa[rki]*Δt*du
                ShallowWaters.axb!(v0,RKaΔt[rki],dv)          #v0 .+= RKa[rki]*Δt*dv
                ShallowWaters.axb!(η0,RKaΔt[rki],dη)          #η0 .+= RKa[rki]*Δt*dη
            end
        end

        if compensated
            # add compensation term to total tendency
            ShallowWaters.axb!(du_sum,-1,du_comp)
            ShallowWaters.axb!(dv_sum,-1,dv_comp)
            ShallowWaters.axb!(dη_sum,-1,dη_comp)

            ShallowWaters.axb!(u0,1,du_sum)   # update prognostic variable with total tendency
            ShallowWaters.axb!(v0,1,dv_sum)
            ShallowWaters.axb!(η0,1,dη_sum)

            ShallowWaters.dambmc!(du_comp,u0,u,du_sum)    # compute new compensation
            ShallowWaters.dambmc!(dv_comp,v0,v,dv_sum)
            ShallowWaters.dambmc!(dη_comp,η0,η,dη_sum)
        end

        ShallowWaters.ghost_points!(u0,v0,η0,S_true)

        # type conversion for mixed precision
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        η0rhs = convert(Diag.PrognosticVarsRHS.η,η0)

        # ADVECTION and CORIOLIS TERMS
        # although included in the tendency of every RK substep,
        # only update every nstep_advcor steps if nstep_advcor > 0
        if dynamics == "nonlinear" && nstep_advcor > 0 && (i % nstep_advcor) == 0
            ShallowWaters.UVfluxes!(u0rhs,v0rhs,η0rhs,Diag,S_true)
            ShallowWaters.advection_coriolis!(u0rhs,v0rhs,η0rhs,Diag,S_true)
        end

        # DIFFUSIVE TERMS - SEMI-IMPLICIT EULER
        # use u0 = u^(n+1) to evaluate tendencies, add to u0 = u^n + rhs
        # evaluate only every nstep_diff time steps
        if (S_true.parameters.i % nstep_diff) == 0
            ShallowWaters.bottom_drag!(u0rhs,v0rhs,η0rhs,Diag,S_true)
            ShallowWaters.diffusion!(u0rhs,v0rhs,Diag,S_true)
            ShallowWaters.add_drag_diff_tendencies!(u0,v0,Diag,S_true)
            ShallowWaters.ghost_points_uv!(u0,v0,S_true)
        end

        # TRACER ADVECTION
        u0rhs = convert(Diag.PrognosticVarsRHS.u,u0)
        v0rhs = convert(Diag.PrognosticVarsRHS.v,v0)
        ShallowWaters.tracer!(i,u0rhs,v0rhs,Prog,Diag,S_true)

        # storing daily states for the "true" values
        if t ∈ 1:225:S_true.grid.nt
            temp1 = ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                ShallowWaters.remove_halo(u,v,η,sst,S_true)...)
            push!(true_states, temp1)
        end

        # generating the data
        if t ∈ S_true.parameters.data_steps
            tempu = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).u)
            tempv = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).v)

            data[:, j] = Float32.([tempu; tempv][Int.(data_spots)] .+ sigma_data .* randn(length(data_spots)))
            j += 1
        end

        if compute_freq

            if t ∈ 10:10:S_true.grid.nt

                tempu = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                    ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).u)
                tempv = vec((ShallowWaters.PrognosticVars{S_true.parameters.Tprog}(
                    ShallowWaters.remove_halo(u,v,η,sst,S_true)...)).v)

                push!(freqpoweru, periodogram(states_adj[end].u; radialavg=true))
                push!(freqpowerv, periodogram(states_adj[end].v; radialavg=true))

            end

        end

        # Copy back from substeps
        copyto!(u,u0)
        copyto!(v,v0)
        copyto!(η,η0)

        t += dtint

    end

    return data, true_states

end


function cpintegrate(chkp, scheme)
    # for l = 1:chkp.S.grid.nt
    S = chkp.S
    # setup
    Diag = S.Diag
    Prog = S.Prog

    @unpack u,v,η,sst = Prog
    @unpack u0,v0,η0 = Diag.RungeKutta
    @unpack u1,v1,η1 = Diag.RungeKutta
    @unpack du,dv,dη = Diag.Tendencies
    @unpack du_sum,dv_sum,dη_sum = Diag.Tendencies
    @unpack du_comp,dv_comp,dη_comp = Diag.Tendencies

    @unpack um,vm = Diag.SemiLagrange

    @unpack dynamics,RKo,RKs,tracer_advection = S.parameters
    @unpack time_scheme,compensated = S.parameters
    @unpack RKaΔt,RKbΔt = S.constants
    @unpack Δt_Δ,Δt_Δs = S.constants

    @unpack nt,dtint = S.grid
    @unpack nstep_advcor,nstep_diff,nadvstep,nadvstep_half = S.grid

    # calculate layer thicknesses for initial conditions
    ShallowWaters.thickness!(Diag.VolumeFluxes.h,η,S.forcing.H)
    ShallowWaters.Ix!(Diag.VolumeFluxes.h_u,Diag.VolumeFluxes.h)
    ShallowWaters.Iy!(Diag.VolumeFluxes.h_v,Diag.VolumeFluxes.h)
    ShallowWaters.Ixy!(Diag.Vorticity.h_q,Diag.VolumeFluxes.h)

    # calculate PV terms for initial conditions
    urhs = convert(Diag.PrognosticVarsRHS.u,u)
    vrhs = convert(Diag.PrognosticVarsRHS.v,v)
    ηrhs = convert(Diag.PrognosticVarsRHS.η,η)

    ShallowWaters.advection_coriolis!(urhs,vrhs,ηrhs,Diag,S)
    ShallowWaters.PVadvection!(Diag,S)

    # propagate initial conditions
    copyto!(u0,u)
    copyto!(v0,v)
    copyto!(η0,η)

    # store initial conditions of sst for relaxation
    copyto!(Diag.SemiLagrange.sst_ref,sst)

    # run integration loop with checkpointing
    S.parameters.i = 1
    S.parameters.j = 1
    # for S.parameters.i = 1:1
    @checkpoint_struct scheme chkp for S.parameters.i = 1:1
        chkp.S.parameters.J += chkp.S.Prog.u[1]# - data[:, S.parameters.j]).^2)
    end

    # return nothing

    return chkp.S.parameters.J

end

function gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

    P = ShallowWaters.Parameter(T=Float32;
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

    S.Prog.u = reshape(param_guess[1:17292], 131, 132)
    S.Prog.v = reshape(param_guess[17293:34584], 132, 131)
    S.Prog.η = reshape(param_guess[34585:end], 130, 130)

    # snaps = Int(floor(sqrt(S.grid.nt)))
    revolve = Revolve{Chkp}()
    data_spots = Int.(data_spots)

    chkp = Chkp(S, data, data_spots)
    dchkp = Enzyme.make_zero(chkp)

    ret=autodiff(
        set_runtime_activity(Enzyme.ReverseWithPrimal),
        cpintegrate,
        Duplicated(chkp, dchkp),
        Const(revolve),
    )
    # MyS = MyModelSetup(
    #     MyParameters(0.0, 1),
    #     MyPrognosticVars(S.Prog.u),
    # )
    # chkp = Chkp(MyS)
    # dchkp = Enzyme.make_zero(chkp)
    # ret=autodiff(
    #     set_runtime_activity(Enzyme.ReverseWithPrimal),
    #     cpintegrate,
    #     Duplicated(chkp, dchkp),
    #     Const(revolve),
    # )
    @show ret
    dS = dchkp.S
    # G = [vec(dS.Prog.u);]
    G = [vec(dS.Prog.u); vec(dS.Prog.v); vec(dS.Prog.η)]
    return S, dS, G

end

function run()

    xu = 30:10:100
    yu = 40:10:100
    Xu = xu' .* ones(length(yu))
    Yu = ones(length(xu))' .* yu

    sigma_data = 0.01
    sigma_initcond = 0.02
    data_steps = 220:220:6733
    data_spotsu = vec((Xu.-1) .* 127 + Yu)
    data_spotsv = vec((Xu.-1) .* 128 + Yu) .+ (128*127)        # just adding the offset of the size of u, otherwise same spatial locations roughly
    data_spots = [data_spotsu; data_spotsv]
    Ndays = 1

    P_pred = ShallowWaters.Parameter(T=Float32;
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
    initpath="./data_files_forkf/128_spinup_noforcing/")

    S_true = ShallowWaters.model_setup(P_pred)

    data, _ = generate_data(S_true, data_spots, sigma_data)

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

    Ndays = copy(P_pred.Ndays)
    data_steps = copy(P_pred.data_steps)

    dS = Enzyme.Compiler.make_zero(S_pred)
    G = zeros(length(dS.Prog.u) + length(dS.Prog.v) + length(dS.Prog.η))
    # S, dS, G = gradient_eval(G, param_guess, nothing, nothing, data_steps, Ndays)
    S, dS, G = gradient_eval(G, param_guess, data, data_spots, data_steps, Ndays)

    return S, dS, G

end

S, dS, G = run()

@show norm(G)
