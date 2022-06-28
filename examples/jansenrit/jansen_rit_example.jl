wdir = @__DIR__
cd(wdir)
out= joinpath(wdir, "out")
for experiment in 1:4
  mkpath(joinpath(out,"experiment"*string(experiment)))
end
mkpath(joinpath(out,"generated"))

using Revise
using LinearAlgebra
using GuidingDiffusion
using Distributions
using Bridge
using Plots
using StaticArrays
using Random
#import Bridge: constdiff


include("plotting.jl")
################################  TESTING  ################################################

S = DE(Vern7())
outdir = joinpath(out,"generated")
include("generatedata.jl")
timegrids = set_timegrids(obs, 0.0005)

#include("bimodalityplot.jl")

verbose = false#true # if true, surpress output written to console
# ESTσ = false

iterations = 2000  #5_00
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

params = [:C, :μ]

temp = 10_000.0   # temperature for exploring chain

Random.seed!(12)

#----------------------------------------------------------------
#for experiment in [3] #1:4

experiment=4
acc =[]

outdir = joinpath(out,"experiment"*string(experiment))

if experiment==1
  swap_allowed = !true # if true, then include exploring chain and try to swap
  Cinit = 25.0
  μinit = 100.0
  # step size par rw par updates
  shortC = longC =0.04
  shortμ= longμ = 0.08
  # pcn pars 
  ρ = 0.9
  ρe = 0.99
end
if experiment==2
  swap_allowed = !true # if true, then include exploring chain and try to swap
  Cinit = 300.0
  μinit = 100.0
  shortC  =0.01
  longC = 0.1
  shortμ = 0.01 
  longμ = 0.1
  # pcn pars 
  ρ = 0.99
  ρe = 0.99
end
if experiment==3
  swap_allowed = true # if true, then include exploring chain and try to swap
  Cinit = 25.0
  μinit = 100.0
  # step size par rw par updates
  shortC  =0.01
  longC = 0.1
  shortμ = 0.01 
  longμ = 0.1
  # pcn pars 
  ρ = 0.995
  ρe = 0.99
end
if experiment==4
  swap_allowed = true # if true, then include exploring chain and try to swap
  Cinit = 300.0
  μinit = 100.0
  # step size par rw par updates
  shortC  =0.01
  longC = 0.1
  shortμ = 0.01 
  longμ = 0.1
  # pcn pars 
  ρ = 0.995
  ρe = 0.99
end

ℙinit = setproperties(ℙ0,  C = Cinit, μ=μinit)




# prior
Π = (A=Exponential(3.25), 
        B=Exponential(22.0),
        C=Exponential(135.0),
        α1=Beta(1.0, 1.0),
        α2=Beta(1.0, 1.0),
        e0=Exponential(5.0),
        v0=Exponential(6.0),
        r=Exponential(0.56),
        μ=Exponential(200.0),
        σ=InverseGamma(0.1, 0.1) )

# random walk step sizes
𝒯 = (A=(short=0.02, long=.1),
B=(short=0.2, long=1.0),
C=(short=shortC, long=longC),      
α1=(short=0.01, long=.1),
α2=(short=0.02, long=.1),
e0=(short=0.2, long=1.0),
v0=(short=0.2, long=1.0),
r=(short=0.2, long=1.0),
μ=(short=shortμ, long=longμ),      
σ=(short=0.2, long=1.0) )

𝒯e = (A=(short=0.02, long=.1),C=(short=1.0, long=50.0), μ=(short=.5, long=4.0))

moves =[ParMove([:C], false, 𝒯, Π), ParMove([:μ], true, 𝒯, Π)]
#moves = [ParMove([:C,:μ], true, 𝒯, Π) ]
swapmove = ParMove([:C,:μ], false, 𝒯, Π)  # should contain all pars in params

if swap_allowed
  movese =[ParMove([:C], false, 𝒯e, Π), ParMove([:μ], true, 𝒯e, Π)]
else
  movese =[]
end

ℙ = ℙinit
ℙe = setproperties(ℙinit, σ = temp)

# initialisation of target chain 
B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids);
Z = Innovations(timegrids, ℙ);
Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = fill(ρ, length(timegrids))
XX, ll = forwardguide(B, ℙ)(x0, Z);


# initialisation of exploring chain 
Be = BackwardFilter(S, ℙe, AuxType, obs, obsvals, timegrids);
Ze = deepcopy(Z)# Innovations(timegrids, ℙe);# deepcopy(Z);
Zeᵒ = deepcopy(Ze)
ρse = fill(ρe, length(timegrids))
XXe, lle = forwardguide(Be, ℙe)(x0, Ze);



θsave = [copy(getpar(params, ℙ))]
XXsave = [copy(XX)]
llsave = [ll]

XXesave = [copy(XXe)]

exploring = [State(x0, copy(Ze), getpar(params,ℙe), copy(lle))] # collection of samples from exploring chain

for i in 1:iterations
  (i % 500 == 0) && println(i)

  # update exploring chain
  if swap_allowed
    for move ∈ movese
      lle, Be, ℙe, acc_ = parupdate!(Be, XXe, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙe, Ze, lle);
      push!(acc, ("exploring_" * string(move.names) , acc_))
    end
    lle, acc_ = pcnupdate!(Be, ℙe, XXe, Zbuffer, Zeᵒ, ρse)(x0, Ze, lle); 
    push!(acc, ("exploring_pcn", acc_))
  end 

  # update target chain   
  for move ∈ moves
    smallworld = rand() > 0.33
    if swap_allowed && !smallworld
      w = sample(exploring)     # randomly choose from samples of exploring chain
      ll, ℙ, acc_ = exploremoveσfixed!(B, Be, ℙe, swapmove, XX, Zᵒ, w; verbose=verbose)(x0, ℙ, Z, ll) 
      push!(acc, ("swap", acc_))
    else
      ll, B, ℙ, acc_ = parupdate!(B, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);
      push!(acc, ("target_" * string(move.names) , acc_))
    end  
  end
  ll, acc_ = pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs)(x0, Z, ll);
  push!(acc, ("target_pcn", acc_))

  # update exploring
  push!(exploring, State(x0, copy(Ze), getpar(params, ℙe), copy(lle)))   

 
  # saving iterates
  push!(θsave, getpar(params, ℙ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))
  (i % 500 == 0) && push!(XXesave, XXe)
  
 # adjust_PCNparameters!(ρs, ρ)
end


include("jansen_rit_makefig.jl")

#end
#----------------------------------------------------------------