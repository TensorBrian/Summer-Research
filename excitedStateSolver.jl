using Random
using LinearAlgebra
using TensorKit
using TensorKitManifolds
using MERAKit
using KrylovKit
# DemoTools holds some utility functions needed by this script, for creating Hamiltonians
# and storing and reading MERAs to/from disk. It exports nothing, so all calls need to be
# qualified as DemoTools.whatever.
include("demo_tools.jl")
using .DemoTools

path1 = "./BinaryIsingExcited2.jld2"
# Pauli matrices

test_mera = DemoTools.load_mera(path1)
H = h = DemoTools.ising_hamiltonian(2.0, 1, symmetry = :Z2, block_size = 2)
V = space(H,1)
print(V)
H = H ⊗ id(V) + id(V) ⊗ H
H = H/2

#print(expect(H, test_mera))


function opScalingData(op, howmany, search_dim)
    f(x) = op * x
    op_space = domain(op)    
    x0 = TensorMap(randn, op_space ← op_space)
    
    return schursolve(f, x0, howmany, :SR, Arnoldi(; krylovdim=search_dim))
end



data = opScalingData(H, 10, 10)
print("\n")
print(dot(data[2][1], data[2][2]))
