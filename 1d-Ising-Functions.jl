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

path = "./demo_result.jld2"

mera = DemoTools.load_mera(path)
h = DemoTools.ising_hamiltonian(; symmetry = :Z2, block_size = 2)

V = ℂ[ℤ₂](0=>1, 1=>1)

#Operators
Z = TensorMap(zeros, Float64, V ← V)
Z.data[ℤ₂(0)] .= 1.0
Z.data[ℤ₂(1)] .= -1.0
eye = id(V)
ZI = Z ⊗ eye
IZ = eye ⊗ Z
ZZ = Z ⊗ Z
II = eye ⊗ eye

#Fusion Mappings
fusionspace = domain(II)
fusetop = isomorphism(fuse(fusionspace), fusionspace)
fusebottom = isomorphism(fusionspace, fuse(fusionspace))

#Fusing Legs of Operators
ZI_fused = fusetop * ZI * fusebottom
IZ_fused = fusetop * IZ * fusebottom
ZZ_fused = fusetop * ZZ * fusebottom
II_fused = fusetop * II * fusebottom

n=2
op1 = ascended_operator(ZI_fused, mera, n)
op2 = ascended_operator(IZ_fused, mera, n)
correlator = op1 ⊗ op2

pars = (;)
print(domain(ZI_fused ⊗ IZ_fused))
print("\n")
print(domain(h))
#print(ZI_fused)
print("\n")
#print(op1)
evalscale = 4
opscale = 2
rho = densitymatrix(mera, evalscale, pars)
op = ascend(IZ_fused ⊗ IZ_fused, mera, evalscale, opscale)

print(domain(rho))
print("\n")
print(domain(op))

value = dot(rho, op)

print("\n")
print(value)
print("\n")

#p = mera.layers[2].isometry
#print(domain(p'))
function __2PCorrelator(Mera, op, q)
    raised_op = op
    
    for i in 1:q
        w = Mera.layers[i].isometry
        raised_op = w' * (II_fused ⊗ raised_op ⊗ II_fused) * w
    end

    return raised_op ⊗ raised_op
end

function __2PCorrelation(Mera, op, q)
    rho = densitymatrix(Mera, q+1, (;))
    correlator = __2PCorrelator(Mera, op, q)
    return dot(rho, correlator)
end



function __4Pcorrelation(Mera, op , q)
    u = Mera.layers[q+1].disentangler
    w = Mera.layers[q+1].isometry
    disentangled_op = u' * __2PCorrelator(Mera, op, q) * u
    pre_op = II_fused ⊗ op ⊗ disentangled_op ⊗ op ⊗ II_fused
    renormalized_op = (w' ⊗ w') * pre_op * (w ⊗ w)

    rho = densitymatrix(Mera, q+2, (;))

    return dot(rho, renormalized_op)
end

function general_2PCorrelator(Mera, op1, op2, n1, n2, l, top)
    newop1 = op1
    newop2 = op2
    local layer
    if l <= num_translayers(mera)+1
        layer = Mera.layers[l]
    else
        layer = Mera.layers[num_translayers(mera)+1]
    end

    if n2>n1
        if n2 - n1 > 5
            if mod(n1, 3) == 0
                newop1 = ascend(op1, )
    
            end

        end

    end
end


print(__2PCorrelation(mera, ZI_fused, 2))
print("\n")
print(__4Pcorrelation(mera, ZZ_fused, 2))
print("\n")
print(num_translayers(mera))
print("\n")

