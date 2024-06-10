function initialize_hyperparameters(y::Vector{Float64}, s::Int64)::Vector{Float64}

    output = fit_SSL(y, s)

    return [output.residuals_variances["ϵ"]; output.residuals_variances["ξ"]; output.residuals_variances["ζ"]; output.residuals_variances["ω"]]
end

function fit_SSM_model(y::Vector{Float64}, s::Int64, initial_hyperparameters::Vector{Float64})

    py"""
    import statsmodels.api as sm
    import numpy as np
    from scipy.stats import norm

    def fit_ssm(y, s, initial_hyperparameters):

        model_components = {'irregular': True, 'level': True, 'trend': True, 'freq_seasonal': [{'period': s}], 
                            'stochastic_level': True, 'stochastic_trend': True, 'stochastic_freq_seasonal': [True]}

        model        = sm.tsa.UnobservedComponents(np.array(y), **model_components)
        fitted_model = model.fit(start_params = initial_hyperparameters, disp = False, maxiter = 1e5) 

        return fitted_model
    """

    return py"fit_ssm"(y, s, initial_hyperparameters)
end

function forecast_SSM_model(fitted_model::Ml, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}} where Ml

    py"""
    import pandas as pd
    import statsmodels.api as sm

    def forecast_ssm(fitted_model, H):

        forec = fitted_model.get_forecast(steps = H)

        return forec.predicted_mean

    def simulate_ssm(fitted_model, H, S):

        simulations = fitted_model.simulate(H, repetitions = S, anchor = 'end')

        return simulations
    """

    prediction = py"forecast_ssm"(fitted_model, H)
    scenarios  = py"simulate_ssm"(fitted_model, H, S)[:, 1, :]

    return prediction, scenarios
end

function get_forecast_SSM(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    initial_hyperparameters = ForecastTester.initialize_hyperparameters(y, s)

    fitted_model          = ForecastTester.fit_SSM_model(y, s, initial_hyperparameters)
    prediction, scenarios = ForecastTester.forecast_SSM_model(fitted_model, H, S)

    return prediction, scenarios
end