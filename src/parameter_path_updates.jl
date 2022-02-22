function setpar(θ, ℙ, move) 
    tup = (; zip(move.names, θ)...) # try copy here
    setproperties(ℙ, tup)
end  

getpar(names_, ℙ) = SVector(ntuple(i -> getproperty(ℙ, names_[i]), length(names_)))# 


function log_prior_ratio(θ, θᵒ, move)
    M = product_distribution(move.prior)
    sum(logpdf(M, θᵒ) - logpdf(M, θ))
end 



# function parameterkernel(θ, tuningpars, s) 
#     shortrange = rand()>s
#     Δ = shortrange ?  rand(MvNormal(tuningpars.short)) : rand(MvNormal(tuningpars.long))
#     θ + Δ
#   end
  
# parameterkernel(tuningpars; s=0.33) = (θ) -> parameterkernel(θ, tuningpars, s) 

# computing log prior ratio
function log_proposal_ratio(θ, θᵒ, param)
    if param ∈ [:α1, :α2]
      return log(θᵒ/θ) + log((1. - θᵒ)/(1. - θ))
    else
      return log(θᵒ/θ)
    end
  end
  
function log_proposal_ratio(θ, θᵒ, move::ParMove)
    out =  0.0
    params = move.names
    for i ∈ eachindex(params)
      out += log_proposal_ratio(θ[i], θᵒ[i], params[i])
    end
    out
  end
  

# define methods for move
propose(move) = (θ) -> move.K(θ)
log_prior_ratio(move) = (θ, θᵒ) -> log_prior_ratio(θ, θᵒ, move)
setpar(move) = (θ, ℙ) -> setpar(θ, ℙ, move) 
getpar(move) = (ℙ) -> getpar(move.names, ℙ) 
log_proposal_ratio(move) = (θ, θᵒ) -> log_proposal_ratio(θ, θᵒ, move)
  


function adjust_PNCparamters!(ρs, ρ; thresh=0.25)
    for i in eachindex(ρs)
        U = rand()
        ρs[i] = ρ * (U<thresh) + (U>=thresh)
    end
end

function pcn!(Zᵒ, Z, Zbuffer, ρs, ℙ)
    noisetype = wienertype(ℙ)
    for i in eachindex(Z.z)
      sample!(Zbuffer.z[i], noisetype)
      Zᵒ.z[i].yy .= ρs[i]*Z.z[i].yy + sqrt(1.0-ρs[i]^2)*Zbuffer.z[i].yy
    end
end
  
  
function copy!(Z1::Innovations, Z2::Innovations)
    for i in eachindex(Z1.z)
        Z1.z[i].yy .= Z2.z[i].yy 
    end
end

function copy!(Z1::Innovations{T}, Z2::Innovations{T}) where {T<:SamplePath}
    for i in eachindex(Z1.z)
        Z1.z[i].yy .= Z2.z[i].yy 
    end
end

import Base.copy
copy(Z::Innovations) = Innovations(deepcopy(Z.z))

function checkstate(w, B, ℙ)
    _, ll = forwardguide(B, ℙ)(x0, w.Z)
    w.ll ==ll, w.ll-ll
end
checkstate(B, ℙ) = (w) -> checkstate(w,B, ℙ)



# par updating, XX is overwritten when accepted
function parupdate!(B, ℙ, x0, Z, ll, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=true)
    accpar_ = false
    θ =  getpar(move)(ℙ)  #move.par(ℙ)
    θᵒ = propose(move)(θ)   
    ℙᵒ = setpar(move)(θᵒ, ℙ)    
    if move.recomputeguidingterm        
        Bᵒ = BackwardFilter(S, ℙᵒ, AuxType, obs, obsvals, timegrids);
    else 
        Bᵒ = B
    end
    XXᵒ, llᵒ = forwardguide(Bᵒ, ℙᵒ)(x0, Z)
    !verbose && printinfo(ll, llᵒ, "par") 

    if log(rand()) < llᵒ-ll + log_prior_ratio(move)(θ, θᵒ) + log_proposal_ratio(move)(θ, θᵒ)
      @. XX = XXᵒ
      ll = llᵒ
      B = Bᵒ
      ℙ = ℙᵒ
      accpar_ = true
      !verbose && print("✓")  
    end
    ll, B, ℙ, accpar_
end

parupdate!(B, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=true)  = (x0, ℙ, Z, ll) -> parupdate!(B, ℙ, x0, Z, ll, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)


# innov updating, XX and Z may get overwritten
function pcnupdate!(B, ℙ, x0, Z, ll, XX, Zbuffer, Zᵒ, ρs; verbose=true)
    accinnov_ = false 
    pcn!(Zᵒ, Z, Zbuffer, ρs, ℙ)
    XXᵒ, llᵒ = forwardguide(B, ℙ)(x0, Zᵒ);
    
    !verbose && printinfo(ll, llᵒ, "pCN") 
    if log(rand()) < llᵒ-ll
        @. XX = XXᵒ
        copy!(Z, Zᵒ)
        ll = llᵒ
        accinnov_ = true
        !verbose && print("✓")  
    end
    ll, accinnov_
end

pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs; verbose=true) = (x0, Z, ll) -> pcnupdate!(B, ℙ, x0, Z, ll, XX, Zbuffer, Zᵒ, ρs; verbose=verbose)


function exploremoveσfixed!(B, ℙ, Be, ℙe, move , x0, Z, ll, XX, Zᵒ, w; verbose=true) # w::State proposal from exploring chain
    accswap_ = false
    # propose from exploring chain in target chain
    θᵒ = copy(w.θ)
    copy!(Zᵒ, w.Z) 
    ℙᵒ = setpar(move)(θᵒ, ℙ) 
    XXᵒ, llᵒ = forwardguide(B, ℙᵒ)(x0, Zᵒ)
        
    # compute log proposalratio, numerator should be πᵗ(θ, Z)
    θ = getpar(move)(ℙ)  
    ℙeprop = setpar(move)(θ, ℙe)
     _, llproposal = forwardguide(Be, ℙeprop)(x0, Z)
    # denominator should be πᵗ(θᵒ, Zᵒ)
    # ℙeᵒ = setpar(move)(θᵒ, ℙe)
    # _, llproposalᵒ = forwardguide(Be, ℙeᵒ)(x0, Zᵒ);
    # llproposalᵒ == w.ll  # should be true
    llproposalᵒ = w.ll


    A = llᵒ - ll + llproposal - llproposalᵒ
    if log(rand()) < A
        println(θᵒ)
        @show llᵒ - ll 
        @show llproposal - llproposalᵒ 
        println()
        
        @. XX = XXᵒ
        copy!(Z, Zᵒ)
        ll = llᵒ
        ℙ = ℙᵒ
        accswap_ = true
        !verbose && print("✓")  
    end
    ll, ℙ, accswap_
end

exploremoveσfixed!(B, Be, ℙe, move, XX, Zᵒ, w; verbose=true) = (x0, ℙ, Z, ll) ->  exploremoveσfixed!(B, ℙ,  Be, ℙe,move, x0, Z, ll, XX, Zᵒ, w; verbose=verbose) # w::State proposal from exploring chain



