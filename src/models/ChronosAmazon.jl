function get_quantiles_chronos(y::Vector{Float64}, H::Int64, model_name::String)

    T = length(y)
    current_date = Date(year(Dates.now()), month(Dates.now()), 1)
    first_date   = current_date - Month(T - 1)

    timestamps = string.(collect(first_date:Month(1):current_date))

    py"""
    import numpy as np
    import pandas as pd
    import torch
    from chronos import ChronosPipeline

    def run_chronos(y, timestamps, H, model_name):

        T = len(y)
        item_vector = np.ones(T)

        data_dict = {'item_id': item_vector.astype(int),
                     'timestamp': timestamps,
                     'target': y}

        df = pd.DataFrame(data_dict)

        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map = "mps"
          )
          
        forecast = pipeline.predict(
            context=torch.tensor(data_dict["target"]),
            prediction_length = H
        )

        dict_forecast = {'mean': np.mean(forecast[0].numpy(), axis = 0),
                        '0.0': np.quantile(forecast[0].numpy(), [0.0], axis = 0),
                        '0.1': np.quantile(forecast[0].numpy(), [0.1], axis = 0),
                        '0.2': np.quantile(forecast[0].numpy(), [0.2], axis = 0),
                        '0.3': np.quantile(forecast[0].numpy(), [0.3], axis = 0),
                        '0.4': np.quantile(forecast[0].numpy(), [0.4], axis = 0),
                        '0.5': np.quantile(forecast[0].numpy(), [0.5], axis = 0),
                        '0.6': np.quantile(forecast[0].numpy(), [0.6], axis = 0),
                        '0.7': np.quantile(forecast[0].numpy(), [0.7], axis = 0),
                        '0.8': np.quantile(forecast[0].numpy(), [0.8], axis = 0),
                        '0.9': np.quantile(forecast[0].numpy(), [0.9], axis = 0),
                        '1.0': np.quantile(forecast[0].numpy(), [1.0], axis = 0),}
        return dict_forecast
    """

    return py"run_chronos"(y, timestamps, H, model_name)
end

function get_forecast_chronos_tiny(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    scenarios       = Matrix{Float64}(undef, H, S)
    quantiles_probs = collect(0.0:0.1:1.0) 

    dict_forecast = ForecastTester.get_quantiles_chronos(y, H, "amazon/chronos-t5-tiny")

    for i in 1:H

        quantiles_values = Vector{Float64}(undef, length(quantiles_probs))
        for (j, q) in enumerate(quantiles_probs)
            quantiles_values[j] = dict_forecast[string(q)][1, :][i]
        end

        inverse_cdf_itp = ForecastTester.interpolate_inverse_cdf(quantiles_probs, quantiles_values)
        scenarios[i, :] = ForecastTester.simulate_from_inverse_cdf(inverse_cdf_itp, S) 
    end

    return dict_forecast["mean"], scenarios
end

function get_forecast_chronos_mini(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    scenarios       = Matrix{Float64}(undef, H, S)
    quantiles_probs = collect(0.0:0.1:1.0) 

    dict_forecast = ForecastTester.get_quantiles_chronos(y, H, "amazon/chronos-t5-mini")

    for i in 1:H

        quantiles_values = Vector{Float64}(undef, length(quantiles_probs))
        for (j, q) in enumerate(quantiles_probs)
            quantiles_values[j] = dict_forecast[string(q)][1, :][i]
        end

        inverse_cdf_itp = ForecastTester.interpolate_inverse_cdf(quantiles_probs, quantiles_values)
        scenarios[i, :] = ForecastTester.simulate_from_inverse_cdf(inverse_cdf_itp, S) 
    end

    return dict_forecast["mean"], scenarios
end

function get_forecast_chronos_small(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    scenarios       = Matrix{Float64}(undef, H, S)
    quantiles_probs = collect(0.0:0.1:1.0) 

    dict_forecast = ForecastTester.get_quantiles_chronos(y, H, "amazon/chronos-t5-small")

    for i in 1:H

        quantiles_values = Vector{Float64}(undef, length(quantiles_probs))
        for (j, q) in enumerate(quantiles_probs)
            quantiles_values[j] = dict_forecast[string(q)][1, :][i]
        end

        inverse_cdf_itp = ForecastTester.interpolate_inverse_cdf(quantiles_probs, quantiles_values)
        scenarios[i, :] = ForecastTester.simulate_from_inverse_cdf(inverse_cdf_itp, S) 
    end

    return dict_forecast["mean"], scenarios
end

function get_forecast_chronos_base(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    scenarios       = Matrix{Float64}(undef, H, S)
    quantiles_probs = collect(0.0:0.1:1.0) 

    dict_forecast = ForecastTester.get_quantiles_chronos(y, H, "amazon/chronos-t5-base")

    for i in 1:H

        quantiles_values = Vector{Float64}(undef, length(quantiles_probs))
        for (j, q) in enumerate(quantiles_probs)
            quantiles_values[j] = dict_forecast[string(q)][1, :][i]
        end

        inverse_cdf_itp = ForecastTester.interpolate_inverse_cdf(quantiles_probs, quantiles_values)
        scenarios[i, :] = ForecastTester.simulate_from_inverse_cdf(inverse_cdf_itp, S) 
    end

    return dict_forecast["mean"], scenarios
end

function get_forecast_chronos_large(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Matrix{Float64}}

    scenarios       = Matrix{Float64}(undef, H, S)
    quantiles_probs = collect(0.0:0.1:1.0) 

    dict_forecast = ForecastTester.get_quantiles_chronos(y, H, "amazon/chronos-t5-large")

    for i in 1:H

        quantiles_values = Vector{Float64}(undef, length(quantiles_probs))
        for (j, q) in enumerate(quantiles_probs)
            quantiles_values[j] = dict_forecast[string(q)][1, :][i]
        end

        inverse_cdf_itp = ForecastTester.interpolate_inverse_cdf(quantiles_probs, quantiles_values)
        scenarios[i, :] = ForecastTester.simulate_from_inverse_cdf(inverse_cdf_itp, S) 
    end

    return dict_forecast["mean"], scenarios
end
