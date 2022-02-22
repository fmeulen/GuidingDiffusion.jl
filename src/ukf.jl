using KalmanFilters
Δt = 0.1
σ_acc_noise = 0.02
σ_meas_noise = 1.0
# Process model
F = [1 Δt Δt^2/2; 0 1 Δt; 0 0 1]
# Process noise covariance
Q = [Δt^2/2; Δt; 1] * [Δt^2/2 Δt 1] * σ_acc_noise^2
# Measurement model
H = [1, 0, 0]'
# Measurement noise covariance
R = σ_meas_noise^2
# Initial state and covariances
x_init = [0.0, 0.0, 0.0]
P_init = [2.5 0.25 0.1; 0.25 2.5 0.2; 0.1 0.2 2.5]
# Take first measurement
measurement = 2.0 + randn()
mu = measurement_update(x_init, P_init, measurement, H, R)
for i = 1:100
    measurement = 2.0 + randn()
    tu = time_update(get_state(mu), get_covariance(mu), F, Q)
    mu = measurement_update(get_state(tu), get_covariance(tu), measurement, H, R)
end

# Non-linear case

# If you define the process model F or the measurement model H as a function (or a functor), the Unscented-Kalman-Filter will be used.

F(x) = x .* [1., 2.]
tu = time_update(x, P, F, Q)



## for Algorithms book

struct UnscentedKalmanFilter 
    μb # mean vector
    Σb # covariance matrix
    λ # spread parameter
end

# helper function
function unscented_transform(μ, Σ, f, λ, ws) 
    n = length(μ)
    Δ = cholesky((n + λ) * Σ).L 
    S = [μ]
    for i in 1:n
        push!(S, μ + Δ[:,i]) 
        push!(S, μ - Δ[:,i])
    end
    S′ = f.(S)
    μ′ = sum(w*s for (w,s) in zip(ws, S′))
    Σ′ = sum(w*(s - μ′)*(s - μ′)' for (w,s) in zip(ws, S′))
    return (μ′, Σ′, S, S′)
end 

# this is the function to call,  𝒫 can be defined using a composite type or a named tuple.
function update(b::UnscentedKalmanFilter, 𝒫, a, o) 
    μb, Σb, λ = b.μb, b.Σb, b.λ
    fT, fO = 𝒫.fT, 𝒫.fO         # this is what 𝒫 needs to contain
    n = length(μb)
    ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]
    # predict
    μp, Σp, Sp, Sp′ = unscented_transform(μb, Σb, s->fT(s,a), λ, ws) 
    Σp += 𝒫.Σs           # this is what 𝒫 needs to contain
    # update
    μo, Σo, So, So′ = unscented_transform(μp, Σp, fO, λ, ws)
    Σo += 𝒫.Σo             # this is what 𝒫 needs to contain
    Σpo = sum(w*(s - μp)*(s′ - μo)' for (w,s,s′) in zip(ws, So, So′)) 
    K = Σpo / Σo
    μb′ = μp + K*(o - μo)
    Σb′ = Σp - K*Σo*K'
    return UnscentedKalmanFilter(μb′, Σb′, λ)
end

struct FilteringProblem{TfT, Tf0, TΣs, TΣo}
    fT::TfT  # state dynamics, fT(s,a) (a is parameter)
    fO::Tf0  # observation dynamics
    Σs::TΣs  # state covariance
    Σo::TΣo  # observations covariance
end

𝒫 = FilteringProblem((s,a) -> [ -atan(a*s[1]), s[2]*s[1]], s-> s[1], [1.2 0.3; 0.3 0.3], 0.5 )
a = 0.2
o = 1.2
b = UnscentedKalmanFilter([0.0, 1.0], [0.5 0.0; 0.0 1.5], 2.0)
update(b, 𝒫, a, o) 


#Algorithm 19.5. The unscented Kalman filter, an extension of the Kalman filter to problems with nonlinear Gaussian dynamics. 
#The current belief is represented by mean μb and covariance Σb. 
# The problem 𝒫 specifies the nonlinear dynamics using the mean transition dynamics function fT and 
# mean observation dynamics function fO. 
# The sigma points used in the unscented transforms are controlled by the spread parameter λ.