wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using Revise
using LinearAlgebra
using GuidingDiffusion
using Distributions
using Bridge
using Plots
using StaticArrays
#import Bridge: constdiff


include("plotting.jl")

################################  TESTING  ################################################

S = DE(Vern7())
swap_allowed = true # if true, then include exploring chain and try to swap

verbose = true # if true, surpress output written to console
# ESTσ = false

iterations = 500  #5_00
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

params = [:C, :μ]

include("generatedata.jl")
timegrids = set_timegrids(obs, 0.0005)

# initialise parameter
Cinit = 15.0
μinit = 100.0
ℙinit = setproperties(ℙ0,  C = Cinit, μ=μinit)

# temperature for exploring chain
temp = 10_000.0 # temperature

# pcn pars 
ρ = 0.99
ρe = 0.99

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
C=(short=0.01, long=.2),
α1=(short=0.01, long=.1),
α2=(short=0.02, long=.1),
e0=(short=0.2, long=1.0),
v0=(short=0.2, long=1.0),
r=(short=0.2, long=1.0),
μ=(short=0.2, long=1.0),
σ=(short=0.2, long=1.0) )

𝒯e = (A=(short=0.02, long=.1),C=(short=10.0, long=50.0), μ=(short=.5, long=2.0))



# moves = repeat([ParMove([:C], false, 𝒯, Π),
#          ParMove([:α1], false, 𝒯, Π)], 3)
# push!(moves, ParMove([:C, :α1], true, 𝒯, Π))

# moves = [ParMove([:C], false, 𝒯, Π),ParMove([:α1], true, 𝒯, Π)]

#params = [:α1]
#moves = [ParMove([:α1], true, 𝒯, Π)]

moves =[ParMove([:C], false, 𝒯, Π), ParMove([:μ], true, 𝒯, Π)]
swapmove = ParMove([:C,:μ], false, 𝒯, Π)  # should contain all pars in params

if swap_allowed
  movese =[ParMove([:C], false, 𝒯e, Π), ParMove([:μ], true, 𝒯e, Π)]
else
  movese =[]
end





ℙ = ℙinit
#ℙe = setproperties(ℙ0,   σ = temp, C = Cinit, μ=μinit)
ℙe = setproperties(ℙinit, σ = temp)


# overwrite, figure out later precisely
# allparnames = params
# allparnamese = params


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

# acceptance rate counters
accinnov = 0
accpar = 0
accinnove = 0
accpare = 0
accmove = 0

accpar_= accpare_ = accinnove_= accinnov_= accmove_ =0


exploring = [State(x0, copy(Ze), getpar(params,ℙe), copy(lle))] # collection of samples from exploring chain


for i in 1:iterations
  (i % 500 == 0) && println(i)
  global accpar_, accpare_,accinnove_, accinnov_, accmove_

  # update exploring chain
  for move ∈ movese
    lle, Be, ℙe, accpare_ = parupdate!(Be, XXe, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙe, Ze, lle);
  end
  if swap_allowed
    lle, accinnove_ = pcnupdate!(Be, ℙe, XXe, Zbuffer, Zeᵒ, ρse)(x0, Ze, lle); 
  end 

  # update target chain   
  for move ∈ moves
    smallworld = rand() > 0.33
    if swap_allowed & !smallworld
      w = sample(exploring)     # randomly choose from samples of exploring chain
      ll, ℙ,  accmove_ = exploremoveσfixed!(B, Be, ℙe, swapmove, XX, Zᵒ, w; verbose=verbose)(x0, ℙ, Z, ll) 
      accpar_ = 0
    else
      ll, B, ℙ, accpar_ = parupdate!(B, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);
      accmove_ =0
    end  
  end
  ll, accinnov_ = pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs)(x0, Z, ll);

  # update exploring
  push!(exploring, State(x0, copy(Ze), getpar(params, ℙe), copy(lle)))   

  # update acceptance counters
  accpar += accpar_; accpare += accpare_; accinnove += accinnove_; accinnov += accinnov_; accmove += accmove_
  
  # saving iterates
  push!(θsave, getpar(params, ℙ))
  push!(llsave, ll)
  (i in subsamples) && push!(XXsave, copy(XX))
  (i % 500 == 0) && push!(XXesave, XXe)
  
 # adjust_PCNparameters!(ρs, ρ)
end


include("jansen_rit_makefig.jl")


