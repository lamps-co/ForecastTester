function fit_ETS_model(y::Vector{Float64}, s::Int64)::Dict

    py"""
    import pandas as pd
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from itertools import product
    from sklearn.metrics import mean_absolute_percentage_error as mape

    def fit_ets(y, s):

        RESTRICTED = [("add", None,  False, "mul"),
              ("add", "add", False, "mul"),
              ("add", "add", True,  "mul"),
              ("mul", "mul", False, "add"),
              ("mul", "mul", True,  "add"),
              ("add", "mul", False, None),
              ("add", "mul", True,  None),
              ("add", "mul", False, "add"),
              ("add", "mul", True,  "add"),
              ("add", "mul", False, "mul"),
              ("add", "mul", True, " mul")]

        MUL_TREND = [("add", "mul", False, "add"),
             ("add", "mul", False, "mul"),
             ("add", "mul", False, None),
             ("add", "mul", True,  "add"),
             ("add", "mul", True,  "mul"),
             ("add", "mul", True,  None),
             ("mul", "mul", False, "add"),
             ("mul", "mul", False, "mul"),
             ("mul", "mul", False, None),
             ("mul", "mul", True,  "add"),
             ("mul", "mul", True,  "mul"),
             ("mul", "mul", True,  None)]

        POSITIVE_OBS = [("add", None, False, None),
                        ("add", None, False, "add"),
                        ("add", "add", False, None),
                        ("add", "add", False, "add"),
                        ("add", "add", True, None),
                        ("add", "add", True, "add")]

        allow_restricted = False
        allow_mul_trend  = False

        T = len(y)

        configs = set()
        if any([y[t] <= 0.0 for t in range(T)]):
            for config in POSITIVE_OBS:
                configs.add(config)
        else:
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
            if not allow_restricted:
                configs = configs.difference(RESTRICTED)
            if not allow_mul_trend:
                configs = configs.difference(MUL_TREND)

        configs = [{"error":            config[0],
                    "trend":            config[1],
                    "damped_trend":     config[2],
                    "seasonal":         config[3],
                    "seasonal_periods": s
                    } for config in configs]

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
