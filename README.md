6/19/25

I currently have 8 files in my repository, those being 2dIsing.jl, demo_tools.jl, finishIt.jl, MERA_functions.jl, test.jl, and usingMERA.jl. 
One should download 

  "TensorKit"          => v"0.9.1"
  "JLD2"               => v"0.1.14"
  "MERAKit"            => v"0.1.0"
  "TensorKitManifolds" => v"0.6.0"
  "Optim"              => v"1.9.3"

  on Julia 1.6 to be able to run my julia files.

  2dIsing:
  Here I'm trying to write the 2d Ising Model on a square lattice projected onto a 1d Lattice. My first attempt at writing this hamiltonian involves
  writing out the nearest neighbor interaction terms and transverse field terms as tensor products of Paulis before summing them. I believe the Hamiltonian
  is correct, but it is quite slow to set up the Hamiltonian for anything more than a 2 by 2 Lattice. That's why for my second formulation I'm attempting to 
  write the Hamiltonian using a tensor network. 

  demo_tools:
  This is a helper function copied over from MERAKit.jl.

  usingMERA:
  This is a demo copied over from MERAKit.jl. Expanding the bonds leads to a bug, so for now I've resorted to commenting it out. (That and 
  I don't think it's thesible to perform such an operation efficiently on a quantum computer....).

  For the most part it appears to work, but after experimenting with different enviornments, sometimes usingMERA outputs a long error message telling
  me that 
  ERROR: LoadError: TypeError: in AbstractTensorMap, in S, expected S<:ElementarySpace, got Type{TensorKit.HomSpace{GradedSpace{Z2Irrep, Tuple{Int64, Int64}}, ProductSpace{GradedSpace{Z2Irrep, Tuple{Int64, Int64}}, 2}, ProductSpace{GradedSpace{Z2Irrep, Tuple{Int64, Int64}}, 5}}}

  I've encountered this error on Julia +1.6.2. On Julia +1.6.7 the issue dissapears. This is worth further investigation, but I'll think about it later.

  finishIt:
  This file is for taking apart usingMERA and disecting it as much as possible, with the endgoal being to further my understanding of MERAKit.jl's innerworkings. 

  test:
  This file is also copied over from MERAKit.jl. I don't fully understand it; it is worth further investigation.

  
