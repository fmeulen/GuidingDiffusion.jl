abstract type Solver end

struct RK4 <: Solver end
struct Vern7direct{T} #<: Solver
    tableau::T
end 
Vern7direct() = Vern7direct(Vern7Tableau())
struct DE{T} <: Solver 
    solvertype::T
end

"""
    parameterkernel(θ, 𝒯 , s, Π) 

    Consstruct kernel for generating a new proposal θ∘ from state θ
    Parameter s is the probability of generating the proposal with widt 'long' paramter from 𝒯
    Π is passed to extract support restrictions and apply transformations to ensure θ∘ is in the support

    Usage:
    K = parameterkernel(𝒯, prior, s)
    θ∘ = K(θ)
"""
function parameterkernel(θ, 𝒯 , s, Π) 
    b = bijector.(Π) # [bijector(x) for x ∈ Π]
    b⁻ = Bijectors.inverse.(b)
    shortrange = rand() > s
    Δ = shortrange ?  rand(MvNormal(𝒯.short)) : rand(MvNormal(𝒯.long))
    #inverse(b)(b(θ) + Δ)
    #@show θ
    [b⁻[i]( b[i](θ[i]) + Δ[i] ) for i in eachindex(θ)]
end
parameterkernel(𝒯, prior, s) = (θ) -> parameterkernel(θ, 𝒯, s, prior) 


"""
    ParMove{Tn, Tkernel, Tp, Tr}

    provide 
    names: vector of Symbols, which are names of pars
    K: kernel for proposing a new value of the parameter
    prior: (product)-distribution 
    recomputeguidingterm: Boolean whether necessary to recompute guiding term with this move
"""
struct ParMove{Tn, Tkernel, Tp, Tr}
  names::Vector{Tn}
  K::Tkernel
  prior::Tp
  recomputeguidingterm::Tr

  Parmove(n::Vector{Tn},K::Tkernel,Π::Tp,rec::Tr) where {Tn, Tkernel, Tp, Tr} =
  new{Tn, Tkernel, Tp, Tr}(n, K, Π, rec)

  function ParMove(names_::Vector{Symbol}, recomputeguidingterm, 𝒯, Π; s=0.33)
    Πsub = [getfield(Π,x) for x in names_] 
    sh = map(x-> getfield(𝒯,x).short, names_)
    lo = map(x-> getfield(𝒯,x).long, names_)
    𝒯sub = (short=sh, long=lo)
    #K = parameterkernel(tp, Πsub, s)
    #K = (θ) -> parameterkernel(θ, 𝒯sub, s, Πsub) 
    K = parameterkernel(𝒯sub, Πsub, s)
    new{eltype(names_), typeof(K), typeof(Πsub), typeof(recomputeguidingterm)}(names_ ,K , Πsub, recomputeguidingterm)
  end  # right now it is recursive
end






struct Innovations{T}
    z::Vector{T}
    
   Innovations(z::Vector{T})  where {T} = new{T}(z)
   function Innovations(timegrids, ℙ) 
      z = [sample(timegrids[i], wienertype(ℙ)) for i in eachindex(timegrids) ]  # innovations process
      new{eltype(z)}(z)
   end
end  


"""
    Observation

    (t,v,L,Σ): at time t we have observations v ~ N(Lx_t, Σ)
    t: time of observation
    h: htransform of the observation 
"""
struct Observation{Tt, Th}
    t::Tt
    h::Th
    Observation(t::Tt,  h::Th) where {Tt,Tv,TL,TΣ,Th} =
        new{Tt,  Th}(t,h)

    function Observation(t::Tt, v::Tv, L::TL, Σ::TΣ) where {Tt, Tv, TL, TΣ}
        h = Htransform(Obs(), v, L, Σ)
        new{Tt, typeof(h)}(t,h)
    end    
end


struct Obs end # for dispatch in Htransform
struct Htransform{TH, TF, TC}
    H::TH
    F::TF
    C::TC

    Htransform(H::TH, F::TF, C::TC) where {TH,TF,TC} = new{TH,TF,TC}(H,F,C)
    
    """
        Htransform(v, L, Σ)

        Convert observation v ~ N(Lx, Σ)
        to triplet (H, F, C), which is of type Htransform
    """
    function Htransform(::Obs, v, L, Σ)
        A = L' * inv(Σ)
        H, F, C = A * L, A*v, -logpdf(Bridge.Gaussian(zero(v), Σ), v) 
        new{typeof(H), typeof(F), typeof(C)}(H, F, C)
    end

end


"""
    Message

        struct containing all information for guiding on a segment (equivalently kernel)
    ℙ̃:  auxiliary NclarDiffusion
    tt: time grid for diffusion (including start and end time)
    H:  H-values on tt
    F:  F values on tt
    C:  C is an additive factor in the loglikelihood

    constructors for solving the backward filtering by numerically approximating ODEs
"""
struct Message{Tℙ̃,TH,TF,TC} 
    ℙ̃::Tℙ̃   
    tt::Vector{Float64}  
    H::Vector{TH}      
    F::Vector{TF}      
    C::TC              
    Message(ℙ̃::Tℙ̃, tt, Ht::Vector{TH}, Ft::Vector{TF}, C::TC) where {Tℙ̃,TH,TF,TC} =
        new{Tℙ̃,TH,TF,TC}(ℙ̃, tt, Ht, Ft, C)

    function Message(D::DE, ℙ̃, tt_, hT::Htransform{TH, TF, TC}) where {TH, TF,TC}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = ode_HFC!(D, ℙ̃, tt, (Ht, Ft), hT)
        new{typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ̃, tt, Ht, Ft, C)
    end

    function Message(::Vern7direct, ℙ̃, tt_, hT::Htransform{TH, TF, TC}) where {TH, TF,TC}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = ode_HFC!(Vern7direct(), ℙ̃, tt, (Ht, Ft), hT)
        new{typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ̃, tt, Ht, Ft, C)
    end

    function Message(::RK4, ℙ̃, tt_, hT::Htransform{TH, TF, TC}) where {TH, TF,TC}
        tt = collect(tt_)
        N = length(tt)
        Ht = zeros(TH, N)
        Ft = zeros(TF, N)
        _, _, C = ode_HFC!(RK4(), ℙ̃, tt, (Ht, Ft), hT)
        new{typeof(ℙ̃), eltype(Ht), eltype(Ft), typeof(C)}(ℙ̃, tt, Ht, Ft, C)
    end
end


function Htransform(M::Message) 
    Htransform(M.H[1], M.F[1], M.C)
end


struct BackwardFilter{T, Th0}
    Ms::Vector{T}
    h0::Th0
    
   BackwardFilter(Ms, h0::Th0)  where {Th0} = new{eltype(Ms), Th0}(Ms, h0)
   
   function BackwardFilter(S, ℙ̃s, obs, timegrids)
       h0, Ms = backwardfiltering(S, obs, timegrids, ℙ̃s)
       new{eltype(Ms), typeof(h0)}(Ms, h0)
   end

end  

struct State{Tx0, TI, Tθ, Tll}
    x0::Tx0
    Z::TI
    θ::Tθ
    ll::Tll
end
  