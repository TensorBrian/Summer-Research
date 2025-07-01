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

path1 = "./BinaryIsingExcited2.jld2"
# Pauli matrices


H = h = DemoTools.ising_hamiltonian(3, 1.0; symmetry = :none, block_size = 2)
V = space(H,1)

H = H ⊗ id(V) + id(V) ⊗ H
H = H/2

mera = DemoTools.load_mera(path1)

GS_densityMatrix = densitymatrix(mera, 1, (;))

H_new = H +2400*3.0839285655175153/0.49950560780908854*GS_densityMatrix
print(expect(H_new, mera))

meratype = BinaryMERA
layers = 2
exact_energy = 4/pi
method = :lbfgs
verbosity = 2
gradient_delta = 1e-3

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
path2 = "./BinaryIsingExcited3.jld2"

m = DemoTools.load_mera(path2)
m1 = minimize_expectation(m ,H_new, pars; finalize! = finalize!)

DemoTools.store_mera(path2, m1)

rho_new = densitymatrix(m1, 1, (;))

print(tr(rho_new*GS_densityMatrix))
print("\n \n")
print(expect(H, m1))
print("\n \n")

new_density = H*GS_densityMatrix
new_density = new_density/tr(new_density)
new_density2 = H*rho_new
new_density2 = H*rho_new/tr(new_density2)

trace_distance = 0.5 * tr(sqrt((new_density - GS_densityMatrix)'*(new_density-GS_densityMatrix)))
print(trace_distance)
print("\n \n")

trace_distance2 = 0.5 * tr(sqrt((new_density2 - rho_new)'*(new_density2-rho_new)))
print(trace_distance2)
print("\n \n")




