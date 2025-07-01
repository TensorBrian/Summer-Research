using Random
using LinearAlgebra
using TensorKit
using TensorKitManifolds
using MERAKit
using KrylovKit
using OptimKit
include("demo_tools.jl")
using .DemoTools
using Printf

path =  "./Binary_MERA_Ising/newising.jld2"

function precondition_tangent(m::GenericMERA, tan::GenericMERA, pars::NamedTuple)
    nt = num_translayers(m)
    tanlayers_prec = ntuple(Val(nt+1)) do i
        l = getlayer(m, i)
        tanl = getlayer(tan, i)
        rho = densitymatrix(m, i+1, pars)
        precondition_tangent(l, tanl, rho)
    end
    tan_prec = GenericMERA(tanlayers_prec)
    return tan_prec
end

function gapped_grad(m, h, pars, gap=0; _finalize! = OptimKit._finalize!)
    function fg(x)
        f = expect(h, x, pars)
        g = gradient(h, x, pars)
        return f, g
    end

    rtrct(args...) = retract(args...; alg = pars[:retraction])
    trnsprt!(args...) = transport!(args...; alg = pars[:transport])
    innr(args...) = inner(args...; metric = pars[:metric])
    scale(vec, beta) = tensorwise_scale(vec, beta)
    add(vec1, vec2, beta) = tensorwise_sum(vec1, scale(vec2, beta))
    linesearch = HagerZhangLineSearch(; ϵ = pars[:ls_epsilon])
    if pars[:precondition]
        precondition(x, g) = precondition_tangent(x, g, pars)
    else
        # The default that does nothing.
        precondition = OptimKit._precondition
    end

    algkwargs = (
        maxiter = pars[:maxiter],
        linesearch = linesearch,
        verbosity = pars[:verbosity],
        gradtol = pars[:gradient_delta],
        #
    )
    local alg
    alg = GradientDescent(; algkwargs...)

    res = optimize_gap(
        fg, m, gap, alg;
        scale! = scale,
        add! = add,
        retract = rtrct,
        inner = innr,
        transport! = trnsprt!,
        isometrictransport = true,
        precondition = precondition,
        _finalize! = OptimKit._finalize!,
    )
    m, expectation, normgrad, fg_num, normgradhistory, breachgap = res
    if pars[:verbosity] > 0
        @info("Gradient optimization done. Expectation = $(expectation).")
    end
    return m, breachgap
end

function optimize_gap(fg, x, gap, alg::GradientDescent;
                  precondition=_precondition,
                  (_finalize!)=OptimKit._finalize!,
                  shouldstop=1000,
                  hasconverged=1e-7,
                  retract=_retract, inner=_inner, (transport!)=_transport!,
                  (scale!)=_scale!, (add!)=_add!,
                  isometrictransport=(transport! == _transport! && inner == _inner))
    t₀ = time()
    verbosity = 2
    f, g = fg(x)
    numfg = 1
    numiter = 0
    innergg = inner(x, g, g)
    normgrad = sqrt(innergg)
    fhistory = [f]
    normgradhistory = [normgrad]
    t = time() - t₀
    _hasconverged = (normgrad <= alg.gradtol)
    _shouldstop = (numiter >= alg.maxiter)
    _breach_gap = (f <= gap)

    # compute here once to define initial value of α in scale-invariant way
    Pg = precondition(x, g)
    normPg = sqrt(inner(x, Pg, Pg))
    α = 1 / (normPg) # initial guess: scale invariant

    numiter = 0
    verbosity >= 2 &&
        @sprintf("GD: initializing with f = %.12f, ‖∇f‖ = %.4e", f, normgrad)
    while !(_hasconverged || _shouldstop)
        # compute new search direction
        Pg = precondition(x, deepcopy(g))
        η = scale!(Pg, -1) # we don't need g or Pg anymore, so we can overwrite it

        # perform line search
        #_xlast[] = x # store result in global variables to debug linesearch failures
        #_glast[] = g
        #_dlast[] = η
        x, f, g, ξ, α, nfg = alg.linesearch(fg, x, η, (f, g);
                                            initialguess=α,
                                            retract=retract, inner=inner)
        numfg += nfg
        numiter += 1
        x, f, g = _finalize!(x, f, g, numiter)
        innergg = inner(x, g, g)
        normgrad = sqrt(innergg)
        push!(fhistory, f)
        push!(normgradhistory, normgrad)
        t = time() - t₀
        _breach_gap = (f <= gap)
        _hasconverged = (normgrad <= gradient_delta)
        _shouldstop = (numiter >= alg.maxiter)


        # check stopping criteria and print info
        if _hasconverged || _shouldstop || _breach_gap
            break
        end
        verbosity >= 2 &&
             @info @sprintf("GD: iter %4d, time %7.2f s: f = %.12f, ‖∇f‖ = %.4e, α = %.2e, nfg = %d",
                           numiter, t, f, normgrad, α, nfg)



        # increase α for next step
        α = 2 * α
    end
    if _hasconverged
        verbosity >= 2 &&
            @info @sprintf("GD: converged after %d iterations and time %.2f s: f = %.12f, ‖∇f‖ = %.4e",
                           numiter, t, f, normgrad)

    else
        verbosity >= 1 &&
            @warn @sprintf("GD: not converged to requested tol after %d iterations and time %.2f s: f = %.12f, ‖∇f‖ = %.4e",
                           numiter, t, f, normgrad)

    end
    history = [fhistory normgradhistory]
    return x, f, g, numfg, history, _breach_gap
end

mera = DemoTools.load_mera(path)
H = h = DemoTools.ising_hamiltonian(2, 1.0; symmetry = :none, block_size = 2)
V = space(H,1)

H = H ⊗ id(V) + id(V) ⊗ H
H = H/2

mera = DemoTools.load_mera(path1)

GS_densityMatrix = densitymatrix(mera, 1, (;))

H_new = H

meratype = BinaryMERA
layers = 10
exact_energy = 4/pi
method = :lbfgs
verbosity = 2
gradient_delta = 1e-7

maxiter_steps = (1000, 500, 500)

checkpoint_frequency = 100
function finalize!(m, energy, g, repnum)
    repnum % checkpoint_frequency != 0 && (return m, energy, g)
    rhoees = densitymatrix_entropies(m)
    scaldims = scalingdimensions(m)
    @info("Energy error: $(energy - exact_energy)")
    @info("Density matrix entropies: $rhoees")
    @info("Scaling dimensions: $scaldims")
    return m, energy, g
end

V_virtual = V
@info("Creating a random MERA with bond dimension $(dim(V_virtual)).")
# The lowest index is of the physical bond dimension, the others are virtual.
Vs = (V, ntuple(x -> V_virtual, layers-1)...)
m = random_MERA(meratype, Float64, Vs)

pars = (
            gradient_delta = gradient_delta,
            maxiter = maxiter_steps[1],
            method = method,
            verbosity = verbosity,
            ls_epsilon = 1e-6,
            precondition = false,
            lbfgs_m = 8,
            vary_disentanglers = true,
            metric = :euclidean,
            retraction = :exp,
            transport = :exp,
    )

#print(gradient(H, m, pars))
local m1
local m
for i in 1:200
    Random.seed!(i)
    m = random_MERA(meratype, Float64, Vs)
    m1, stop = gapped_grad(m ,H, pars, -1; )
    
    if !stop
        print("\n\n")
        print("Seed: ")
        print(i)
        print("\n\n")
        break
    end
end

