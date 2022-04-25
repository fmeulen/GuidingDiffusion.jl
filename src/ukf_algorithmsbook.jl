using LinearAlgebra
using Distributions
using Plots

struct FilteringProblem{Tfs, Tfo, TΣs, TΣo}
    fs::Tfs  # state dynamics, fT(s,a) (a is parameter)
    fo::Tfo  # observation dynamics
    Σs::TΣs  # state covariance
    Σo::TΣo  # observation covariance
end

struct UnscentedKalmanFilter{Tμ, TΣ, Tλ}
    μ::Tμ # mean vector
    Σ::TΣ # covariance matrix
    λ::Tλ # spread parameter in unscented transform
end


"""
    unscented_transform(μ, Σ, f, λ, ws) 

    Compute unscented transoform of f(X), where X ~ N(μ,Σ) using weights ws and parameter λ 
"""
function unscented_transform(μ, Σ, f, λ, ws) 
    n = length(μ)
    Δ = cholesky((n + λ) *  Symmetric(Σ)).L 
    S = [μ]
    for i in 1:n
        push!(S, μ + Δ[:,i]) 
        push!(S, μ - Δ[:,i])
    end
    Sᵒ = f.(S)
    μᵒ = sum(w*s for (w,s) in zip(ws, Sᵒ))
    Σᵒ = sum(w*(s - μᵒ)*(s - μᵒ)' for (w,s) in zip(ws, Sᵒ))
    
    return (μᵒ, Σᵒ, S, Sᵒ)
end 


"""
    update(b::UnscentedKalmanFilter, 𝒫::FilteringProblem, a, o) 

    Perform one update step of unscented Kalman filter with newly incoming observation o and parameter value a
"""
function update(u::UnscentedKalmanFilter, 𝒫::FilteringProblem, a, o) 
    μ, Σ, λ = u.μ, u.Σ, u.λ
    fs, fo = 𝒫.fs, 𝒫.fo         
    n = length(μ)
    ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]
    μp, Σp, _, _ = unscented_transform(μ, Σ, s->fs(s,a), λ, ws)   # predict (computes m_k- and P_k- in Sarkka notation, algorithm 5.14)
    Σp += 𝒫.Σs           
    μo, Σo, So, Soᵒ = unscented_transform(μp, Σp, fo, λ, ws)    # update
    Σo += 𝒫.Σo             
    Σpo = sum(w*(s - μp)*(sᵒ - μo)' for (w,s,sᵒ) in zip(ws, So, Soᵒ)) # matrix C_k in Sarkka notation
    K = Σpo / Σo
    μᵒ = μp + K*(o - μo)
    Σᵒ = Σp - K*Σo*K'
    return UnscentedKalmanFilter(μᵒ, Σᵒ, λ)
end


## testing (example 5.1 in Sarkka, swinging pendulum)
δ = 0.01
fs = (s,a) -> [s[1] + δ*s[2], s[2] - 9.8*sin(s[1]) * δ]
fo = s -> [sin(s[1])]
Σs = [δ^3/3 δ^2/2; δ^2/2 δ]
Σo = [0.2]
𝒫 = FilteringProblem(fs, fo, Σs, Σo)

a = 1.01  # parameter value (fixed, not used in this example)


# generate some data
s = [0.0, 0.5] #rand(2)  # initial state
o = 𝒫.fo(s) + rand(MvNormal(𝒫.Σo))
ss= [s]
oo = [o]
for i in 1:250
    s = 𝒫.fs(s,a) + rand(MvNormal(𝒫.Σs))
    o = 𝒫.fo(s) + rand(MvNormal(𝒫.Σo))
    push!(ss, s)
    push!(oo, o)
end


# UKF
# Initialisation 
μinit = rand(2) # [0.0, 0.0] # prior mean vector
Σinit = [5.0 0.0; 0.0 5.0] # prior covariance matrix
λ = 2.0 # spread parameter
u = UnscentedKalmanFilter(μinit, Σinit, λ)
𝒰 = [u]
for o in oo
#    println(o)
    u = update(u, 𝒫, a, o)
    push!(𝒰, u)
end

𝒰

# plotting 
plot(first.(ss),label="true latent state component 1")
scatter!(first.(oo), label="observations")
#plot!(last.(ss),label="true latent state component 2")

filtered = map(x -> x.μ, 𝒰[2:end])
plot!(first.(filtered), label="filtered component 1", linewidth=3)
#plot!(last.(filtered), label="filtered component 2", linewidth=3)




