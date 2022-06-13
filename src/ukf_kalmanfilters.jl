using KalmanFilters
using LinearAlgebra
using Distributions
using Plots




function ukf(observations, (m_init, P_init), (F, Q), (H, R), ℙ)
    mu = measurement_update(m_init, P_init, observations[1], H, R)
    mu_save = [mu]
    for o in observations[2:end]
        tu = time_update(get_state(mu), get_covariance(mu), s->F(s,ℙ), Q(ℙ))
        mu = measurement_update(get_state(tu), get_covariance(tu), o, H, R)
        push!(mu_save, mu)
    end
    mu_save
end


################ data dynamics ################
struct Pendulum
    δ::Float64
    g::Float64
end



# Process model
F(s, ℙ) = [s[1] + ℙ.δ*s[2], s[2] - ℙ.g *sin(s[1]) * ℙ.δ] #- [0.2*s[1], 0.3]
# Process noise covariance
Q(ℙ) = [ℙ.δ^3/3 ℙ.δ^2/2; ℙ.δ^2/2 ℙ.δ]
# Measurement model
H(s) = [sin(s[1])]
Hmat = [1.0 0.0] # Matrix([1.0]')
# Measurement noise covariance
R = Matrix([0.1]')


################ generate some data ################
ℙ0 = Pendulum(0.0001, 9.8) 
s = [0.0, 0.5] 
o = H(s) + rand(MvNormal(R))
ss= [s]
oo = [o]
for i in 1:400_000
    s = F(s,ℙ0) + rand(MvNormal(Q(ℙ0)))
    o = H(s) + rand(MvNormal(R))
    #o = Hmat*s + rand(MvNormal(R))
    push!(ss, s)
    push!(oo, o)
end

# subsamples

oo =oo[1:200:end]
ss = ss[1:200:end]

################ ukf ################
ℙ = Pendulum(200*ℙ0.δ, 9.8) 

# Initial state and covariances
m_init = [0.0, 0.0]
P_init = [5.0 0.0 ; 0.0 5.0]

mu_save = ukf(oo, (m_init, P_init), (F,Q), (H,R), ℙ)

p0 = plot(size=(500,200))
scatter!(p0, first.(oo), label="obs", markeralpha=0.2, markersize=2)
plot!(p0,first.(ss),label="x1 true", linewidth=2, legend_position=:outerright)
plot!(p0, first.(get_state.(mu_save)), linewidth=2, linestyle=:dot, label="UKF")
p0

########### now including parameter (g) estimation ################

Faug(s, ℙ) = [s[1] + ℙ.δ*s[2], s[2] - s[3] *sin(s[1]) * ℙ.δ, s[3]] 
# Process noise covariance
c = 0.8 # artifical noise sd
Qaug(ℙ) = [ℙ.δ^3/3 ℙ.δ^2/2 0.0 ; ℙ.δ^2/2 ℙ.δ 0.0 ; 0.0 0.0 c^2]

# Initial state and covariances
m_init = [0.0, 0.0, 5.0]
P_init = [5.0 0.0 0.0 ; 0.0 5.0 0.0; 0.0 0.0 5.0]

mu_aug_save = ukf(oo, (m_init, P_init), (Faug,Qaug), (H,R), ℙ)

p1 = plot(size=(500,200))
scatter!(p1, first.(oo), label="obs", markeralpha=0.2, markersize=2)
plot!(p1,first.(ss),label="x1 true", linewidth=2, legend_position=:outerright)
plot!(p1, first.(get_state.(mu_aug_save)), linewidth=2, linestyle=:dot, label="UKF")

g_ = last.(get_state.(mu_aug_save))
p2 = plot(size=(500,200))
plot!(p2, g_,label="UKF", linewidth=2, legend_position=:outerright)
hline!(p2, [ℙ.g], label="g true")

@show mean(g_)
lay = @layout [a; b;c]
plot(p0, p1, p2, layout=lay)

# savefig(p0,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum.pdf")
# savefig(p1,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_withpar1.pdf")
# savefig(p2,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_withpar2.pdf")


# savefig(p0,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_long.pdf")
# savefig(p1,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_withpar1_long.pdf")
# savefig(p2,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_withpar2_long.pdf")

savefig(p0,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_verylong.pdf")
savefig(p1,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_withpar1_verylong.pdf")
savefig(p2,"/Users/frankvandermeulen/Dropbox/Jansen-Rit model/tex/ukf_pendulum_withpar2_verylong.pdf")



################ application for JR model as in Kuhlmann ################
# x[7]= b, x[8]=B, x[9]=μ

using GuidingDiffusion
using StaticArrays
using Bridge

# generate data outside this file 

struct JansenRitDiffusion_UKF{T} 
    A::T    # excitatory: maximal amplitudes of the post-synaptic potentials (in millivolts)
    a::T    # excitatory: characteristic for delays of the synaptic transmission
    #B::T    # inhibitory: maximal amplitudes of the post-synaptic potentials (in millivolts)
    #b::T    # inhibitory: characteristic for delays of the synaptic transmission
    C::T    # connectivity constant (C1=C)   
    α1::T   # C2=α1C
    α2::T   # C3=C4=α2C
    e0::T   # half of the maximum firing rate of neurons families
    v0::T   # firing threshold (excitability of the populations)
    r::T    # slope of the sigmoid at v0;
    μ::T    # mean of average firing rate
    σ::T    # A * a * sd of average firing rate (d p(t) = μ dt + σ d Wₜ) (stochasticity accounting for a non specific background activity)
    η::T     # sd of artifical parameter evolution
    δ::T    # discretisation step size
end


JansenRitDiffusion_UKF(P::JansenRitDiffusion, η, δ) = JansenRitDiffusion_UKF(P.A, P.a, P.C, P.α1, P.α2, P.e0, P.v0, P.r, P.μ, P.σ, η, δ)
sigm(x, P::Union{JansenRitDiffusion, JansenRitDiffusionAux, JansenRitDiffusion_UKF}) = 2.0P.e0 / (1.0 + exp(P.r*(P.v0 - x)))

# generate data and return ℙ0 and δ
θ0 =[3.25, 100.0, 22.0, 50.0, 135.0, 0.8, 0.25, 5.0, 6.0, 0.56, 2.0, 2000.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
ℙ0 = JansenRitDiffusion(θ0...)


T = 5.0 
x00 = @SVector zeros(6)
W = sample((-1.0):0.0001:T, wienertype(ℙ0))                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = Bridge.solve(Bridge.Euler(), x00, W, ℙ0)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]

skipobs = 40# I took 400  all the time
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> x[2]-x[3], Xf.yy[1:skipobs:end])
pF = plot_all(ℙ0,  Xf, obstimes, obsvals)

@show  [ℙ0.b,  ℙ0.B, ℙ0.μ]
P = JansenRitDiffusion_UKF(ℙ0, 0.0001, obstimes[2]-obstimes[1])


F(x,P) =     [x[1] + P.δ * x[4], 
                x[2] + P.δ *x[5], 
                x[3] + P.δ *x[6],
                x[4] + P.δ * (P.A*P.a*(sigm(x[2] - x[3], P)) - 2P.a*x[4] - P.a*P.a*x[1]),
                x[5] + P.δ * (P.a * P.A * x[9] + P.A*P.a*P.α1*P.C*sigm(P.C*x[1], P) - 2P.a*x[5] - P.a*P.a*x[2]),
                x[6] + P.δ *x[8]*x[7]*( P.α2*P.C*sigm(P.α2*P.C*x[1], P)) - 2*x[7]*x[6] - x[7]*x[7]*x[3], 
                x[7], x[8], x[9]]


Qmat(P) = Matrix(Diagonal([0.0, 0.0, 0.0, 0.0, P.δ * P.σ^2, 0.0, P.η^2, P.η^2, P.η^2]) ) #+ fill(0.001, 9,9)

Hmat = fill(0.0, 1,9); Hmat[2]=1.0; Hmat[3]=-1

R = Matrix([0.01]')



###################### UKF
# Initial state and covariances
θinit = [ℙ0.b,  ℙ0.B, ℙ0.μ]
m_init = vcat(x0, θinit)
P_init = Matrix(Diagonal(vcat(fill(0.1, 6), fill(1.0,3)) ))

mu = measurement_update(m_init, P_init, obsvals[1], Hmat, R)
tu = time_update(get_state(mu), get_covariance(mu), s->F(s,P), Qmat(P))
mu = measurement_update(m_init, P_init, obsvals[2], Hmat, R)
tu = time_update(get_state(mu), get_covariance(mu), s->F(s,P), Qmat(P))


mu_save = ukf(obsvals, (m_init, P_init), (F,Qmat), (Hmat,R), P)

p1 = plot(map(x->get_state(x)[7], mu_save),label="b", linewidth=2)
p2 = plot(map(x->get_state(x)[8], mu_save),label="B", linewidth=2)
p3 = plot(map(x->get_state(x)[9], mu_save),label="μ", linewidth=2)
l = @layout [a b c]
plot(p1,p2,p3, layout=l)