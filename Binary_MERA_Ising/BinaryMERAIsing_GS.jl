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
# Pauli matrices


H = h = DemoTools.ising_hamiltonian(; symmetry = :none, block_size = 2)
V = space(H,1)

H = H ⊗ id(V) + id(V) ⊗ H
H = H/2

meratype = BinaryMERA
layers = 1

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