"""
    Get the forecast using a structural model.

    Args:
        y::Vector{Float64}: Time series data
        s::Int64: Seasonality
        H::Int64: Forecast horizon
        S::Int64: Number of scenarios for simulation

    Returns:
        prediction::Vector{Float64}: Forecasted values
        scenarios::Matrix{Flaot64}: Simulated scenarios
"""
function get_forecas_SS(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    if s == 1
        model = StateSpaceModels.UnobservedComponents(y; trend = "local linear trend")
    else
        model = StateSpaceModels.UnobservedComponents(y; trend = "local linear trend", seasonal = "stochastic $s")
    end
    StateSpaceModels.fit!(model)
    
    prediction = StateSpaceModels.forecast(model, H).expected_value
    scenarios  = StateSpaceModels.simulate_scenarios(model, H, S)

    return vcat(prediction...), scenarios[:, 1, :]
end