function fit_SSL(y::Vector{Float64}, s::Int64)

    return StateSpaceLearning.fit_model(y; s = s, α = 0.1, outlier = false, stabilize_ζ = 12)
end


