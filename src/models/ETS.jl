function fit_ETS_model(y::Vector{Float64}, s::Int64)::Dict

    py"""
    import pandas as pd
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from itertools import product
    from sklearn.metrics import mean_absolute_percentage_error as mape

    # CONFIRMAR A LÓGICA COM O JOÃO LUCAS
    def fit_ets(y, s):

        configs = set()
        for config in product(["add", "mul"],
                            ["add", "mul"],
                            [False, True],
                            ["add", "mul", None]):
            configs.add(config)
        for config in product(["add", "mul"],
                            [None],
                            [False],
                            ["add", "mul", None]):
            configs.add(config)

        configs = [{"error":            config[0],
                    "trend":            config[1],
                    "damped_trend":     config[2],
                    "seasonal":         config[3],
                    "seasonal_periods": s
                    } for config in configs]

        # PERGUNTAR PARA O JOÃO LUCAS O QUE SIGNIFICA A SINTAXE ** DO PYTHON
        best_config = configs[0]
        best_ets = ETSModel(
            pd.Series(y), **best_config).fit(
            full_output = False, disp = False)

        for config in configs[1:-1]:
            ets = ETSModel(
                pd.Series(y), **config).fit(
                full_output = False, disp = False)
            if mape(y,
                    ets.fittedvalues) > 1. and \
            mape(y,
                    best_ets.fittedvalues) < 1.:
                continue
            if (best_ets.aic * ets.aic > 0. and ets.aic < best_ets.aic) or \
            (best_ets.aic < 0.0 and ets.aic > 0.0):
                best_config = config
                best_ets = ets

        ets_fit = list(best_ets.fittedvalues)

        results_dict = {"fitted_values": ets_fit,
                        "fitted_model": best_ets,
                        "ets_config": best_config}

        return results_dict
    """

    return py"fit_ets"(y, s)
end

function forecast_ETS_model(results_dict::Dict, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    py"""
    import pandas as pd
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    def forecast_ets(results_dict, H):

        start_idx = len(results_dict["fitted_values"])

        forec = list(results_dict["fitted_model"].get_prediction(
                    start = start_idx,
                    end = start_idx + H - 1
        ).summary_frame()["mean"])

        return forec

    def simulate_ets(results_dict, H, S):

        simulations = results_dict["fitted_model"].simulate(H, repetitions = S, anchor = 'end')
        return simulations.values
    """

    prediction = py"forecast_ets"(results_dict, H)
    scenarios  = py"simulate_ets"(results_dict, H, S)

    return prediction, scenarios
end

function get_forecast_ETS(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    results_dict          = ForecastTester.fit_ETS_model(y, s)
    prediction, scenarios = ForecastTester.forecast_ETS_model(results_dict, H, S)

    return prediction, scenarios
end
