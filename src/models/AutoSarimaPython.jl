"""
    Calculates the AIC value of a Sarima model.

    Args:
        residuals::Vector{Float64}: Time-series of the model residuals
        K::Int64: Number of parameters of the model

    Returns:
        Float64: The AIC value
"""
function aic_model(residuals::Vector{Float64} , K::Int64)::Float64
    return -2 * sum(logpdf.(Normal(0, std(residuals)), residuals)) + 2 * K
end

"""
    Return the Sarima model with the best AIC value.

    Args:
        y::Vector{Float64}: Time series data
        fitted_models::Vector: Vector of dictionaries with the fitted models during the auto-arima process

    Returns:
        Dict: A dictionary with the model with the best AIC
"""
function get_best_aic_model(y::Vector{Float64}, fitted_models::Vector)::Dict
    aic_vec = Float64[]
    min_length = Inf
    for i in eachindex(fitted_models)
        min_length = min(min_length, length(fitted_models[i]["fit_in_sample"]))
    end
    min_length = Int64(min_length)
    for i in eachindex(fitted_models)
        residuals = y[end-min_length+1:end] - fitted_models[i]["fit_in_sample"][end-min_length+1:end]
        intercept = fitted_models[i]["intercept"]
        
        K = intercept ? 1 : 0
        for key in keys(fitted_models[i]["sarima_config"])
            # if key != "D" && key != "d"
            K += fitted_models[i]["sarima_config"][key]
            # end
        end
        push!(aic_vec, aic_model(residuals , K))
    end
    return fitted_models[argmin(aic_vec)]
end

"""
    Runs the Auto-Sarima process.

    Args:
        y::Vector{Float64}: Time series data
        s::Int64: Seasonality

    Returns:
        Dict: A dictionary with the model with the best AIC
"""
function fit_autosarima_model(y::Vector{Fl}, s::Int64)::Dict where {Fl} 
    py"""
    import numpy as np
    import statistics
    from statsforecast.models import AutoARIMA
    import statsmodels.api as sm
    import warnings
    """

    py"""
    def auto_sarimax(y, s):        
        y           = np.array(y)
        end_index   = len(y)
        start_index = end_index - 1095 if end_index >= 1095 else 0

        sliced_y    = y[start_index:end_index]
        fitted_models = []

        range_d = range(0,2) 
        range_D = range(0,2)

        for d in range_d:
            for D in range_D:
                fitted_model = AutoARIMA(d = d, D = D, 
                                season_length = s,
                                method = 'CSS-ML',
                                seasonal = True,
                                allowmean = True,
                                stepwise = True,
                                trace = False).fit(sliced_y)
                if 'intercept' in list(fitted_model.model_['coef'].keys()):
                    trend = 'c'
                    selected_intercept = True
                else:
                    trend = None
                    selected_intercept = False
                selected_order = fitted_model.model_['arma']
                order = (selected_order[0], selected_order[5], selected_order[1])
                seasonal_order = (selected_order[2], selected_order[6], selected_order[3], selected_order[4])
                fitted = fit_sarimax_from_auto(y, d, D, s, True, True, order, seasonal_order, trend)
                fit_in_sample, aic, bic, aicc, sarima_config = get_results(fitted, order, seasonal_order, trend)
                fitted_models.append({'dict_fitted_model': fitted,
                                      'fit_in_sample': fit_in_sample,
                                      'aic': aic,
                                      'bic': bic,
                                      'aicc': aicc,
                                      'sarima_config': sarima_config,
                                      'intercept': selected_intercept})
        return fitted_models

    def fit_sarimax_from_auto(y, d, D, m, has_seasonal, intercept, order, seasonal_order, trend):
        y = np.array(y)
        if seasonal_order[3] == 1:
            fitted_model = sm.tsa.SARIMAX(y, order = order, 
                                                    trend = trend).fit(method = "nm", maxiter = 1000)
            with warnings.catch_warnings(record=True) as warning_list:
                fitted_model.summary()
            if len(warning_list) > 0:
                fitted_model = sm.tsa.SARIMAX(y, order = order, 
                                            seasonal_order = seasonal_order,
                                            trend = trend).fit(method = "bfgs", maxiter = 1000)
        else:
            fitted_model = sm.tsa.SARIMAX(y, order = order, 
                                             seasonal_order = seasonal_order,
                                             trend = trend).fit(method = "nm", maxiter = 1000)
            with warnings.catch_warnings(record=True) as warning_list:
                fitted_model.summary()
            if len(warning_list) > 0:
                fitted_model = sm.tsa.SARIMAX(y, order = order, 
                                            seasonal_order = seasonal_order,
                                            trend = trend).fit(method = "bfgs", maxiter = 1000)
        return fitted_model

    def get_results(fitted_model, order, seasonal_order, trend):
        # --- Get results --- #  
        number_of_hyperparameters  = fitted_model.df_model
        hyperparameters            = fitted_model.params
        hyperparameters_pvalue     = fitted_model.pvalues

        selected_p = order[0]
        selected_d = order[1]
        selected_q = order[2]
        selected_P = seasonal_order[0]
        selected_D = seasonal_order[1]
        selected_Q = seasonal_order[2]
        selected_m = seasonal_order[3]
        
        sarima_config = {}
        sarima_config['p'] = selected_p
        sarima_config['d'] = selected_d
        sarima_config['q'] = selected_q
        sarima_config['P'] = selected_P
        sarima_config['D'] = selected_D
        sarima_config['Q'] = selected_Q

        if selected_m - 1 > 0:
            seasonal_entries = selected_m - 1
        else:
            seasonal_entries = 0
        start_results = selected_d + (seasonal_entries + 1) * selected_D + 1 + max(selected_p + selected_m * selected_P, selected_q + selected_m * selected_Q + 1) - 1 

        # Information Criteria
        aic  = fitted_model.aic
        bic  = fitted_model.bic
        aicc = fitted_model.aicc
        
        # Fit in-sample
        fit_in_sample = fitted_model.fittedvalues
        
        return fit_in_sample[start_results:], aic, bic, aicc, sarima_config
    """
    fitted_models = py"auto_sarimax"(y, s)
    return get_best_aic_model(y, fitted_models)
end

"""
    Runs the forecast and simulation of a Sarima model.

    Args:
        fitted_model::Ml: Fitted model
        H::Int64: Forecast horizon
        S::Int64: Number of scenarios for simulation

    Returns:
        Tuple{Vector{Float64}, Matrix{Float64}}: Point forecast and simulated scenarios
        
"""
function forecast_autosarima_model(fitted_model::Ml, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}} where {Ml}

    py"""
    import numpy as np
    import statsmodels.api as sm
    
    def forecast_sarima_model(fitted_model, steps_ahead):

        forec = fitted_model.get_forecast(steps = steps_ahead)

        return forec.predicted_mean

    def simulate_sarima_model(fitted_model, steps_ahead, num_scenarios):

        simulations = fitted_model.simulate(steps_ahead, repetitions = num_scenarios, anchor = 'end')

        return simulations
    """

    prediction = py"forecast_sarima_model"(fitted_model["dict_fitted_model"], H)
    scenarios  = py"simulate_sarima_model"(fitted_model["dict_fitted_model"], H, S)

    return prediction, scenarios[:, 1, :]
end

"""
    Get the forecast using a Auto-Sarima model implemented in Python.

    Args:
        y::Vector{Float64}: Time series data
        s::Int64: Seasonality
        H::Int64: Forecast horizon
        S::Int64: Number of scenarios for simulation

    Returns:
        prediction::Vector{Float64}: Forecasted values
        scenarios::Matrix{Flaot64}: Simulated scenarios
"""
function get_forecast_sarima(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    fitted_model = ForecastTester.fit_autosarima_model(y, s)
    prediction, scenarios = ForecastTester.forecast_autosarima_model(fitted_model, H, S)

    return prediction, scenarios
end