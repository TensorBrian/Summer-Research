using Random
using LinearAlgebra
using TensorKit
using TensorKitManifolds
using MERAKit

path = "./demo_result.jld2"
include("demo_tools.jl")
using .DemoTools

optimMERA = DemoTools.load_mera(path)

entropy = densitymatrix_entropies(optimMERA)

#print(entropy)
#print(h)
 V = ℂ[ℤ₂](0=>1, 1=>1)
# Pauli Z
Z = TensorMap(zeros, Float64, V ← V)
Z.data[ℤ₂(0)] .= 1.0
Z.data[ℤ₂(1)] .= -1.0
eye = id(V)
ZI = Z ⊗ eye
IZ = eye ⊗ Z
# Pauli XX
XX = TensorMap(zeros, Float64, V ⊗ V ← V ⊗ V)
XX.data[ℤ₂(0)] .= [0.0 1.0; 1.0 0.0]
XX.data[ℤ₂(1)] .= [0.0 1.0; 1.0 0.0]
pre_obs = eye ⊗ Z
fusionspace = domain(pre_obs)
fusetop = isomorphism(fuse(fusionspace), fusionspace)
fusebottom = isomorphism(fusionspace, fuse(fusionspace))
op = fusetop * pre_obs * fusebottom

pre_obs2 = Z ⊗ Z
fusetop = isomorphism(fuse(fusionspace), fusionspace)
fusebottom = isomorphism(fusionspace, fuse(fusionspace))
op2 = fusetop * pre_obs2 * fusebottom

opFinal = op ⊗ op2

#print(expect(opFinal, optimMERA))


Ising = DemoTools.ising_hamiltonian(; symmetry = :Z2, block_size = 2)

print(expect(Ising, optimMERA))