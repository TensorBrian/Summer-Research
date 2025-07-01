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

path = "./scaleInvariantResults.jld2"

mera = DemoTools.load_mera(path)


#Operators
V = ℂ^2
X = TensorMap(zeros, Float64, V ← V)
Z = TensorMap(zeros, Float64, V ← V)
eye = id(V)
X.data .= [0.0 1.0; 1.0 0.0]
Z.data .= [1.0 0.0; 0.0 -1.0]
XX = X ⊗ X
XI = X ⊗ eye
IX = eye ⊗ X
ZI = Z ⊗ eye
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
XI_fused = fusetop * XI * fusebottom
IX_fused = fusetop * IX * fusebottom
XX_fused = fusetop * XX * fusebottom
II_fused = fusetop * II * fusebottom



#p = mera.layers[2].isometry
#print(domain(p'))
function __1PScalingOp(Mera, op, l)

    local layer
    if l < num_translayers(Mera)
        layer = Mera.layers[l]
    else
        layer = Mera.layers[num_translayers(Mera)+1]
    end
    
    w = layer.isometry

    raised_op = w' * (II_fused ⊗ op ⊗ II_fused) * w
    
    return raised_op
end

function __2PCorrelator(Mera, op, q)
    raised_op = op
    
    for i in 1:q
        raised_op = __1PScalingOp(Mera, raised_op, i)
        
    end

    return raised_op ⊗ raised_op
end

function __2PCorrelation(Mera, op, q)
    local rho
    if q < num_translayers(Mera)
        rho = densitymatrix(Mera, q+1, (;))
    else
        rho = densitymatrix(Mera, num_translayers(Mera)+1, (;))
    end
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

function __1siteScalingData(Mera, l, howmany)
    f(x) = __1PScalingOp(Mera, x, l)
    x0 = ZZ_fused
    
    return eigsolve(f, x0, howmany, :LM)
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

data = __1siteScalingData(mera, 2, 7)

vecs = data[1]

#print(__2PCorrelation(mera, ZI_fused, 2))
#print("\n")
#print(__4Pcorrelation(mera, ZZ_fused, 2))
#print("\n")
#print(num_translayers(mera))
#print("\n")


L = 20

x = range(0, L, length = L+1)
x_out = 3 .^ x
C_x = []
C_y = []
C_z = []
local MinMin
MinMin = 100
for i in 0:L
    temp = __2PCorrelation(mera, XI_fused, i) #- expect(ZI_fused ⊗ II_fused, mera)^2
    push!(C_x, temp)
    temp = __2PCorrelation(mera, XI_fused*ZI_fused, i)
    push!(C_y, temp)
    #Questionable correction for logplot purposes
    temp = -0.0003708+__2PCorrelation(mera, ZI_fused, i)- expect(ZI_fused ⊗ II_fused, mera)^2 
    push!(C_z, temp)
    if temp < MinMin
        global MinMin = temp
    end
end

C_x_out = log10.(C_x)
C_y_out = log10.(C_y)
C_z_out = log10.(C_z)
#x_out = log10.(x_out)

#Δy = y[L] - y[1]
#Δx = x_out[L] - x_out[1]

#print(Δy/Δx)
print("\n")
#print(expect(II_fused ⊗ IZ_fused, mera))

#plot(x, [C_x C_y C_z], label=["X-Correlator" "Y-Correlator" "Z-Correlator"])
print(C_z[10])
plot(x_out, [C_x C_y C_z], label=["X-Correlation" "Y-Correlation" "Z-Correlation"])
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
xlims!(1e+0, 1e+10)
ylims!(1e-8, 1e+0)
title!("Log-log Correlations")
xlabel!("Lattice Spacing")
ylabel!("Log Correlation")
