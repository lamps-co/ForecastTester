function fit_SSL(y::Vector{Float64}, s::Int64)

    return StateSpaceLearning.fit_model(y; s = s, α = 0.1, outlier = false, stabilize_ζ = 12)
end

function interpolate_inverse_cdf(quantiles_probs::Vector{Float64}, quantiles_values::Vector{Float64})

    return extrapolate(interpolate((quantiles_probs,), quantiles_values, Gridded(Linear())), Flat())
end

function simulate_from_inverse_cdf(inverse_cdf::Itp, num_scenarios::Int64) where {Itp}

    scenarios = Vector{Float64}(undef, num_scenarios)
    dist = Uniform(0, 1)

    Random.seed!(12345)
    for s in 1:num_scenarios
        p = rand(dist)
        scenarios[s] = inverse_cdf(p)
    end

    return scenarios
end
