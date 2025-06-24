using Random
using LinearAlgebra
using TensorKit
using TensorKitManifolds
using MERAKit
# DemoTools holds some utility functions needed by this script, for creating Hamiltonians
# and storing and reading MERAs to/from disk. It exports nothing, so all calls need to be
# qualified as DemoTools.whatever.
include("demo_tools.jl")
using .DemoTools

path = "./newising.jld2"

V = ℂ^2
# Pauli matrices
function singleIsing(J, h)
    X = TensorMap(zeros, Float64, V ← V)
    Z = TensorMap(zeros, Float64, V ← V)
    eye = id(V)
    X.data .= [0.0 1.0; 1.0 0.0]
    Z.data .= [1.0 0.0; 0.0 -1.0]
    XX = X ⊗ X
    ZI = Z ⊗ eye
    IZ = eye ⊗ Z
    H = J*XX + h/2 * (ZI+IZ)

    H_out = H ⊗ eye + eye ⊗ H
    H_out = H_out/2

    return H_out
end

H = singleIsing(1,1)

meratype = BinaryMERA
layers = 7

method = :lbfgs
verbosity = 2
gradient_delta = 1e-9

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
            verbosity = verbosity
    )
m = minimize_expectation(m ,H, pars; finalize! = finalize!)
DemoTools.store_mera(path, m)