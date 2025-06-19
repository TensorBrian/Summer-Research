using Random
using LinearAlgebra
using TensorKit
using TensorKitManifolds
using MERAKit
using MPSKitModels

path = "./demo_result.jld2"
include("demo_tools.jl")
using .DemoTools

optimMERA = DemoTools.load_mera(path)

entropy = densitymatrix_entropies(optimMERA)

#print(entropy)
#print(h)
V = ℂ^2
# Pauli Z
Z_float = Float64[1.0 0; 0.0 -1.0]
Z = TensorMap(Z_float, V ← V)
eye = id(V)

# Pauli x
X_float = Float64[0 1.0; 1.0 0]
X = TensorMap(X_float, V ← V)





function XXTwist(i, m, l2 ,c2)
    if (m == 0) 
        if i == 0
            out_OP = X ⊗ X ⊗ id(V^(4*l2*c2 - 2))
        else 
            out_OP = id(V^i) ⊗ X ⊗  X ⊗ id(V^(4*l2*c2-2-i))
        end

    else
        if i == 0
            out_OP = X ⊗ id(V^m) ⊗  X ⊗ id(V^(4*l2*c2 - 2 - m))
        else 
            out_OP = id(V^i) ⊗ X ⊗ id(V^m) ⊗  X ⊗ id(V^(4*l2*c2-2-m-i))
        end
    end
    return out_OP
end

print(domain(XXTwist(0, 0, 2, 2)))

#This is one way of constructing the Ising Model Hamiltonian in 2-D. It is rather slow, however.
function Transverse_ising_1(l = 1, c = 1, coef_1 = 1, coef_2 = 1)
    V_ising = V^(4*l*c)
    print(V_ising)
    local_I = TensorMap(zeros, Float64, V_ising ← V_ising)
    for i in 0:(4*l*c-1)
        if mod(i, 2) == 0
            if i+1 < 4*l*c
                local_I += coef_1*XXTwist(i, 0, l, c)
            end
        else 
            if i+3 < 4*l*c
                local_I += coef_1*XXTwist(i,2, l, c)
            end
        end
        if mod(i, 4) < 2
            if i+2 < 4*l*c
                local_I += coef_1*XXTwist(i,1, l, c)
            end
        else
            if i+4*l - 2 < 4*l*c
                local_I += coef_1*XXTwist(i,4*l -3, l, c)
            end
        end
   
        if i == 0
            local_I += coef_2 * Z ⊗ id(V^(4*l*c-1))
        else
            local_I += coef_2 * id(V^i) ⊗ Z ⊗ id(V^(4*l*c-i-1))
        end
    end

    return local_I
end
#Hamilton_Ising = Transverse_ising_1(2, 2, 1, 1)
print("\n")
#print(domain(Hamilton_Ising))
print("\n")

#Now I will construct the Ising Model Hamiltonian as a PEPO. I hope it's faster

function Transverse_ising_2(l = 1, c=1, coef_1 = 1, coef_2 = 1)
    T_array = Float64[0 for i in 1:2, j in 1:2, k in 1:3, l in 1:3, m in 1:3, n in 1:3]

    #Identity Terms
    T_array[:, :, 1, 1, 1, 1] = Float64[1.0 0; 0.0 1.0]
    T_array[:, :, 2, 2, 2, 2] = Float64[1.0 0; 0.0 1.0]

    #Transverse Field
    T_array[:, :, 2, 1, 1, 1] = coef_2Float64[1.0 0; 0.0 -1.0]

    #Interaction 1
    T_array[:, :, 2, 1, 3, 1] = coef_1*Float64[0 1.0; 1.0 0.0]
    T_array[:, :, 2, 1, 1, 3] = coef_1*Float64[0 1.0; 1.0 0.0]

    #Interaction 2
    T_array[:, :, 3, 1, 1, 1] = Float64[0 -1.0; -1.0 0.0]
    T_array[:, :, 1, 3, 1, 1] = Float64[0 -1.0; -1.0 0.0]


    T = TensorMap(T_array, ℂ^2 ← ℂ^2⊗ (ℂ^3)^4)

    return T
end