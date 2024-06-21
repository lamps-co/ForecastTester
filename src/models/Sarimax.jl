"""
    Get the forecast using a structural model.

    Args:
        y::Vector{Float64}: Time series data
        s::Int64: Seasonality
        H::Int64: Forecast horizon
        S::Int64: Number of scenarios for simulation

    Returns:
        prediction::Vector{Float64}: Forecasted values
        scenarios::Matrix{Float64}: Simulated scenarios
"""
function get_forecast_SARIMAX(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}
    df = DataFrame(y = y)
    dataset = Sarimax.loadDataset(df)
    model = Sarimax.auto(dataset;seasonality = s, seasonalIntegrationTest="ocsb",assertStationarity=true, assertInvertibility=true)
    Sarimax.predict!(model; stepsAhead=H)
    scenarios = Sarimax.simulate(model, H, S)

    prediction::Vector{Float64}  = TimeSeries.values(model.forecast)
    simulatedScenarios::Matrix{Float64}  = permutedims(hcat(values(scenarios)...))'

    return prediction, simulatedScenarios
end