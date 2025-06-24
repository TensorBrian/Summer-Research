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
include("demo_tools.jl")
using .DemoTools

path = "./newising.jld2"

#Pauli Matrices
V = ℂ^2
# Pauli matrices
X = TensorMap(zeros, Float64, V ← V)
Z = TensorMap(zeros, Float64, V ← V)
eye = id(V)
X.data .= [0.0 1.0; 1.0 0.0]
Z.data .= [1.0 0.0; 0.0 -1.0]

ZII = Z ⊗ eye ⊗ eye
IZI = eye ⊗ Z ⊗ eye
IIZ = eye ⊗ eye ⊗ Z
III = eye ⊗ eye ⊗ eye

mera = DemoTools.load_mera(path)

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
    
    local max_layer
    if q <= max_layers
        max_layer = layers[q]
    else
        max_layer = layers[max_layers]
    end

    for i in 1:q
        if i <= max_layers
            newop1 = ascend_left(newop1, layers[i])
            newop2 = ascend_right(newop2, layers[i])
        else
            newop1 = ascend_left(newop1, layers[max_layers])
            newop2 = ascend_right(newop2, layers[max_layers])
        end
    end

    outputop = even_ascend_block(newop1 ⊗ newop2, 6, max_layer)
    outputop = even_ascend_block(outputop, 4, max_layer)

    return outputop
end

function even_ascend_block(op, len, layer)
    if mod(len, 2) != 0
        throw("len not divisible by 2")
    end

    w = layer.isometry
    u = layer.disentangler
    Vspace = space(u, 1)
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

    u_layer = eye ⊗ u_layer ⊗ eye

    output = (u_layer * w_layer)' * (eye ⊗ op ⊗ eye) * (u_layer * w_layer)
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

print(domain(even_ascend_block(X ⊗ X ⊗ X ⊗ X ⊗ X ⊗ X, 6, mera.layers[2])))
print("\n")
print(domain(__2PBinaryCorrelator(mera, 2, ZII, IZI)))
print("\n")
print(__2PBinaryCorrelation(mera, 8, ZII, IZI))

L=10

#pathplot = "./correlationPLOT.jld2"
x_in = range(1, L, length = L)
x_out = 2 .^ x_in
x_out = 4 .* x_out
x_out = log.(x_out)
y_out = []
for i in 1:L
    temp = __2PBinaryCorrelation(mera, i, III, III)
    push!(y_out, log((temp)))
end

print("\n")
print(y_out[7])
print("\n")
print(y_out[8])

plot(x_out, y_out)

print(scalingdimensions(mera))
