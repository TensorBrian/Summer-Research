using Random
using LinearAlgebra
using TensorKit
using TensorKitManifolds
using MERAKit
using KrylovKit
using Plots

# DemoTools holds some utility functions needed by this script, for creating Hamiltonians
# and storing and reading MERAs to/from disk. It exports nothing, so all calls need to be
# qualified as DemoTools.whatever.
include("C:/Users/brian/OneDrive/GitHub/Summer-Research/demo_tools.jl")
using .DemoTools

path = "./newising.jld2"

#Pauli Matrices
V = ℂ^2
# Pauli Z
Z = TensorMap(zeros, Float64, V ← V)
X = TensorMap(zeros, Float64, V ← V)
X.data .= [0.0 1.0; 1.0 0.0]
Z.data .= [1.0 0.0; 0.0 -1.0]
eye = id(V)
ZI = Z ⊗ eye
IZ = eye ⊗ Z
ZZ = Z ⊗ Z
II = eye ⊗ eye
XI = X ⊗ eye
IX = eye ⊗ X
XX = X ⊗ X
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

mera = DemoTools.load_mera(path)
#h = DemoTools.ising_hamiltonian(; symmetry=:Z2, block_size=2)
#VSpace_h = space(h, 1)
#h = h ⊗ id(VSpace_h) + id(VSpace_h) ⊗ h
#h = h/2

print("\n")
print("\n")
#print(expect(h, mera))
print("\n")
print("\n")

#mera = remove_symmetry(mera)
#h = DemoTools.ising_hamiltonian(; symmetry=:none, block_size=2)
#VSpace_h = space(h, 1)
#h = h ⊗ id(VSpace_h) + id(VSpace_h) ⊗ h
#h = h/2
#mera = minimize_expectation()


print("\n")
print("\n")
#print(expect(h, mera))
print("\n")
print("\n")


##Binary MERA 2-point correlator
##
##Assume that the left operator and right operator 
##are placed such that the left operator always ascends left
##and the right operator always ascends right before 
##colliding. Assume also that upon colliding, the left and
##right operator are perfectly adjacent. __2PBinaryCorrelator
##returns the renormalized/ascended correlator in such a case.

##Params:
##q: # of ascensions
##op1: leftmost operator
##op2: rightmost operator

function __2PBinaryCorrelator(mera, q, op1, op2)
    newop1 = op1
    newop2 = op2
    layers = mera.layers
    max_layers = num_translayers(mera)+1
    
    max_layer = layers[max_layers]

    for i in 1:q
        if i <= max_layers
            newop1 = ascend_left(newop1, layers[i])
            newop2 = ascend_right(newop2, layers[i])
        else
            newop1 = ascend_left(newop1, layers[max_layers])
            newop2 = ascend_right(newop2, layers[max_layers])
        end
    end
    print("\n")
    print(domain(op1))
    print("\n")
    print(domain(op2))
    print("\n")
    local outputop
    if q+2 <= max_layers
        print(1)
        print("\n")
        outputop = even_ascend_block_6(newop1, newop2, layers[q+1])
        outputop = even_ascend_block(outputop, 4, layers[q+2])
    else
        print(0)
        print("\n")
        outputop = even_ascend_block_6(newop1, newop2, max_layer)
        outputop = even_ascend_block(outputop, 4, max_layer)
    end

    return outputop
end

function even_ascend_block(op, len, layer)
    if mod(len, 2) != 0
        throw("len not divisible by 2")
    end

    w = layer.isometry
    u = layer.disentangler
    Vspace = space(u, 1)
    #print(Vspace)
    
    eye = id(Vspace)

    w_layer = w
    u_layer = u

    w_len = Int.(floor(len/2))+1
    u_len = Int.(floor(len/2))

    for i in 1:w_len-1
        w_layer = w_layer ⊗ w
    end

    for j in 1:u_len-1
        u_layer = u_layer ⊗ u
    end
    #print("\n")
    #print(domain(w_layer))
    #print("\n")
    #print(domain(u_layer))
    #print("\n")
    u_layer = eye ⊗ u_layer ⊗ eye

    output = w_layer' * u_layer' * (eye ⊗ op ⊗ eye) * (u_layer * w_layer)

    return output
end

function even_ascend_block_6(op1, op2, layer)

    w = layer.isometry
    u = layer.disentangler
    Vspace = space(u, 1)
    #print(Vspace)
    
    eye = id(Vspace)
    #print("\n")
    op1 = (u ⊗ eye)' * op1 * (u ⊗ eye)
    op2 = (eye ⊗ u)' * op2 * (eye ⊗ u)
    #print(codomain( eye ⊗ op1))
    #print("\n")
    #print(domain((w ⊗ eye ⊗ eye)' * ( eye ⊗ op1)))
    op1 = (w ⊗ eye ⊗ eye)' * (eye ⊗ op1)
    op1 = op1 * (w ⊗ eye ⊗ eye)
    op2 = (eye ⊗ eye ⊗ w)' * (op2 ⊗ eye) * (eye ⊗ eye ⊗ w)

    newop = op1 ⊗ op2
    newop = (eye ⊗ eye ⊗ u ⊗ eye ⊗ eye)' * newop * (eye ⊗ eye ⊗ u ⊗ eye ⊗ eye)
    newop = (eye ⊗ w ⊗ w ⊗ eye)' * newop * (eye ⊗ w ⊗ w ⊗ eye)

    return newop
end

function ascend_left(op, layer)
    u, w = layer
    @planar(
        scaled_op[-100 -200 -300; -400 -500 -600] :=
        w[5 6; -400] * w[9 8; -500] * w[16 15; -600] *
        u[1 2; 6 9] * u[10 12; 8 16] *
        op[3 4 14; 1 2 10] *
        u'[7 13; 3 4] * u'[11 17; 14 12] *
        w'[-100; 5 7] * w'[-200; 13 11] * w'[-300; 17 15]
    )
    return scaled_op
end

function ascend_right(op, layer)
    u, w = layer
    @planar(
        scaled_op[-100 -200 -300; -400 -500 -600] :=
        w[15 16; -400] * w[8 9; -500] * w[6 5; -600] *
        u[12 10; 16 8] * u[1 2; 9 6] *
        op[14 3 4; 10 1 2] *
        u'[17 11; 12 14] * u'[13 7; 3 4] *
        w'[-100; 15 17] * w'[-200; 11 13] * w'[-300; 7 5]
    )
    return scaled_op
end

function __2PBinaryCorrelation(mera, q, op1, op2)
    local rho
    if q > num_translayers(mera)-1
        rho = densitymatrix(mera, (num_translayers(mera)+1), (;))
    else
        rho = densitymatrix(mera, q+2, (;))
    end
    op = __2PBinaryCorrelator(mera, q, op1, op2)
    return dot(rho, op)
end

function ascend_left_scalingData(Mera, l, howmany)
    layer = Mera.layers[l]
    f(x) = ascend_left(x, layer)
    x0 = Z ⊗ Z ⊗ Z
    
    return eigsolve(f, x0, howmany, :LM)
end

function ascend_right_scalingData(Mera, l, howmany)
    layer = Mera.layers[l]
    f(x) = ascend_right(x, layer)
    x0 = Z ⊗ Z ⊗ Z
    
    return eigsolve(f, x0, howmany, :LM)
end

function H(n)
    if n <1
        return 0
    elseif  n == 1
        return 1
    else 
        return factorial(convert(Int32, n-1)) * H(convert(Int32, n-1))
    end
end

print("\n")
#print(expect(ZII, mera))

print("\n")
print("\n")


x = range(0, 10, length=11)
x_pre = 2 .^ x
x_pre = 4 .* x_pre
x_out = 2 .* x_pre #convert.(Int64, x_pre)


z = []
x = []
y = []

z_exact = 1.0 ./((4 .* (x_out .^ 2)) .- 1) .* 1.0/(pi^2)
x_exact =  [0.25 * exp(0.25) * 2^(1.0/12) * 1.2824^(-3) ] .* (x_out .^ (-0.25) )#- x_out .^(-2.25) ./ 64) 
y_exact = 1.0 ./(4 .* x_out .^ 2 .- 1) .* x_exact
print(z_exact)
#print(ascend_left(ZI_fused⊗II_fused⊗II_fused, mera.layers[1]))
print("\n")
print(num_translayers(mera))
print("\n")
op1 = ZI_fused ⊗ II_fused ⊗ II_fused
op2 = II_fused ⊗ ZI_fused ⊗ II_fused
op3 = XI_fused ⊗ II_fused ⊗ II_fused
op4 = II_fused ⊗ XI_fused ⊗ II_fused
op5 = op1*op3
op6 = op2*op4


##Here, "adjust" shifts the C_z correlator before adding it to the z array. 
##Since the GS we calculated is not necessarily exact, we can 
adjust = .00#010#9
for i in 0:10
    
    temp = __2PBinaryCorrelation(mera, i, op1, op2) - expect(op1, mera)^2 - adjust
    push!(z, temp)
    temp2 = __2PBinaryCorrelation(mera, i, op3, op4)
    push!(x, temp2)
    temp3 = __2PBinaryCorrelation(mera, i, op5, op6)
    push!(y, temp3)
end
#print(even_ascend_block(input, 6, mera.layers[1]))
#y = log.(y)
print(y)
#print(__2PBinaryCorrelation(mera, 1, op1, op2))
#print(z[33])
print("\n")
plot(x_out, [x_exact y_exact z_exact], label = ["Exact X-Correlation" "Exact Y-Correlation" "Exact Z-Correlation"])
scatter!(x_out, [x y z], label=["Calculated X-Correlation" "Calculated Y-Correlation" "Calculated Z-Correlation"])
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
xlims!(1e+0, 1e+10)
ylims!(1e-8, 1e+0)
title!("Log-log Correlations")
xlabel!("Lattice Spacing")
ylabel!("Correlation")

