"""
    Get the forecast using the seasonal naive method.

    Args:
        y::Vector{Float64}: Time series data
        s::Int64: Seasonality
        H::Int64: Forecast horizon
        S::Int64: Number of scenarios for simulation

    Returns:
        prediction::Vector{Float64}: Forecasted values
        scenarios::Nothing
"""
function get_forecast_naive(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Nothing}

    y_copy = deepcopy(y)
    T = length(y)

    prediction = Vector{Float64}(undef, H)
    for i in 1:H
        prediction[i] = y_copy[T - s + i]
        push!(y_copy, prediction[i])
    end
    
    return prediction, nothing
end