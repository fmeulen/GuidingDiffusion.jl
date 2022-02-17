vectorise(P,ν, C) = vcat(SVector(P), ν, SVector(C))

"""
static_accessor_HFc(u::SVector, ::Val{T}) where T
Access data stored in the container `u` so that it matches the shapes of H,F,c
and points to the correct points in `u`. `T` is the dimension of the stochastic
process.

implemented by M. Mider in GuidedProposals.jl
"""
function static_accessor_HFc(u::K, ::Val{T}) where {K<:Union{SVector,MVector},T}
    Hidx = SVector{T*T,Int64}(1:T*T)
    Fidx = SVector{T,Int64}((T*T+1):(T*T+T))
    reshape(u[Hidx], Size(T,T)), u[Fidx], u[T*T+T+1]
end

dHFC_DE(y, ℙ̃, s) = dHFC(s, y, ℙ̃)

function ode_HFC!(D::DE, ℙ̃, tt, (Ht, Ft), hT)
    access = Val{}(dim(ℙ̃))
    TP = typeof(hT.H); Tν= typeof(hT.F); Tc = typeof(hT.C)
    saved_values = SavedValues(Float64, Tuple{TP,Tν,Tc})

    yT = vectorise(hT.H, hT.F, hT.C)
    prob = ODEProblem{false}(
            dHFC_DE,                    # increment
            yT,                         # starting val
            (tt[end], tt[1]),           # time interval
            ℙ̃)                          # parameter
    
    callback = SavingCallback(
        (u,t,integrator) -> static_accessor_HFc(u, access),
        saved_values;
        saveat=reverse(tt), 
        tdir=-1
    )
    integrator = init(
        prob,
        D.solvertype,
        callback=callback,
        save_everystep=false, # to prevent wasting memory allocations
    )
#    sol = DifferentialEquations.solve!(integrator)   # s
 
    DifferentialEquations.solve!(integrator)   # s
    ss = saved_values.saveval  # these are in reversed order, so C is obtained from the last index and for Ht and Ft we need to reverse
    for i in eachindex(ss)
        Ht[end-i+1] = ss[i][1]
        Ft[end-i+1] = ss[i][2]
    end
    C = ss[end][3]  
    Ht, Ft, C
end

function dHFC(s, y, ℙ̃)
    access = Val{}(dim(ℙ̃))  #
    #access = Val{6}()
    H, F, _ = static_accessor_HFc(y, access)
    _B, _β, _σ, _a = Bridge.B(s, ℙ̃), Bridge.β(s, ℙ̃), Bridge.σ(s, ℙ̃), Bridge.a(s, ℙ̃)

    dH = (- (_B' * H)  - (H * _B) + Bridge.outer( H * _σ))
    dF  = (- (_B' * F) + H * (_a * F + _β) )
    dC =  (dot(_β, F) + 0.5*dot(F, _a, F) - 0.5*tr( (H* (_a))))
    #U = (F' * _σ) ;  dC =  (dot(_β, F) + 0.5*dot(U,U) - 0.5*tr( (H* (_a))))
    vectorise(dH, dF, dC)
end



function ode_HFC!(::RK4, ℙ̃, t, (Ht, Ft), hT)
    access = Val{}(dim(ℙ̃))  ##access = Val{6}()   
    y = vectorise(hT.H, hT.F, hT.C)
    Ht[end], Ft[end], C =  static_accessor_HFc(y, access)
    
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = kernelrk4(dHFC, t[i+1], y, dt, ℙ̃)
        Ht[i], Ft[i], C = static_accessor_HFc(y, access)
    end
    Ht, Ft, C
end

function ode_HFC!(S::Vern7direct, ℙ̃, t, (Ht, Ft), hT)
    access = Val{}(dim(ℙ̃))  ##access = Val{6}()   
    y = vectorise(hT.H, hT.F, hT.C)
    Ht[end], Ft[end], C =  static_accessor_HFc(y, access)
    
    for i in length(t)-1:-1:1
        dt = t[i] - t[i+1]
        y = vern7(dHFC, t[i+1], y, dt, ℙ̃, S.tableau)
        Ht[i], Ft[i], C = static_accessor_HFc(y, access)
    end
    Ht, Ft, C
end






"""
    fusion_HFC(h1, h2)

    returns added characteristics that correspond to fusion in (H,F,C)-parametrisation
"""
fusion_HFC(h1, h2) = Htransform(h1.H + h2.H, h1.F + h2.F, h1.C + h2.C)


############# here rewrite with handwritten vern function
function backwardfiltering(S,obs, timegrids, ℙ̃s)
    hT = obs[end].h 
    M = Message(S, ℙ̃s[end], timegrids[end], hT) 
    Ms = [M]
    nseg = length(timegrids)
    for i in nseg:-1:2
        hT = fusion_HFC(Htransform(M), obs[i].h)
        M = Message(S, ℙ̃s[i-1], timegrids[i-1], hT) 
        pushfirst!(Ms, M)   
    end
    hT = Htransform(M)
#    hT = fusion_HFC(Htransform(M), obs[1].h) # only if x0 is not fully observed
    hT, Ms
end



function backwardfiltering!(S, Ms, obs) ##FIXME
    n = length(Ms)
    hT = obs[end].h
    for i in n:-1:1
        ode_HFC!(S, Ms[i].ℙ̃, Ms[i].tt, (Ms[i].H, Ms[i].F), hT) #FIXME  S=DE(Vern7())
        hT = fusion_HFC(Htransform(Ms[i]), obs[i].h)
    end
    hT
end

