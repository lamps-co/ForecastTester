"""
    Calculate the sMAPE metric for a given forecast and test set.

    Args:
        y_test::Vector{Fl}: Vector with the test set values
        y_forecast::Vector{Fl}: Vector with the forecasted values

    Returns:
        Float64: sMAPE metric
"""
function sMAPE(y_test::Vector{Fl}, y_forecast::Vector{Fl})::Float64 where {Fl}
    H = length(y_test)

    denominator = abs.(y_test) + abs.(y_forecast)
    return (200/H)*sum(abs(y_test[i] - y_forecast[i])/(denominator[i]) for i in 1:H)
end
 
"""
    Calculate the MASE metric for a given forecast, train set, test set and seasonality.

    Args:
        y_train::Vector{Fl}: Vector with the training set values
        y_test::Vector{Fl}: Vector with the test set values
        y_forecast::Vector{Fl}: Vector with the forecasted values
        s::Int64: Seasonality of the time series

    Returns:
        Float64: MASE metric
"""
function MASE(y_train::Vector{Fl}, y_test::Vector{Fl}, y_forecast::Vector{Fl}, s::Int64)::Float64 where {Fl}
    T = length(y_train)
    H = length(y_test)

    numerator   = (1/H) * sum(abs(y_test[i] - y_forecast[i]) for i in 1:H)
    denominator = (1/(T - s)) * sum(abs(y_train[j] - y_train[j - s]) for j in s+1:T)
    return numerator/denominator
end

"""
    Calculate the OWA metric for a given MASE and sMAPE metrics and their respective benchmark results.

    Args:
        mase::Float64: MASE metric
        mase_benchmark::Float64: MASE benchmark metric
        smape::Float64: sMAPE metric
        smape_benchmark::Float64: sMAPE benchmark metric

    Returns:
        Float64: OWA metric
"""
function OWA(mase::Float64, mase_benchmark::Float64, smape::Float64, smape_benchmark::Float64)::Float64
    return 0.5*(((mase)/(mase_benchmark))+((smape)/(smape_benchmark)))
end 

"""
    Calculate the MSE metric for a given forecast and test set.

    Args:
        y_test::Vector{Fl}: Vector with the test set values
        y_forecast::Vector{Fl}: Vector with the forecasted values

    Returns:
        Float64: MSE metric
"""
function MSE(y_test::Vector{Fl}, y_forecast::Vector{Fl})::Float64 where {Fl}
    return mean((y_test - y_forecast).^2)
end

"""
    Calculate the MAE metric for a given forecast and test set.

    Args:
        y_test::Vector{Fl}: Vector with the test set values
        y_forecast::Vector{Fl}: Vector with the forecasted values

    Returns:
        Float64: MAE metric
"""
function MAE(y_test::Vector{Fl}, y_forecast::Vector{Fl})::Float64 where {Fl}
    return mean(abs.(y_test - y_forecast))
end

"""
    Calculate the MAPE metric for a given forecast and test set.

    Args:
        y_test::Vector{Fl}: Vector with the test set values
        y_forecast::Vector{Fl}: Vector with the forecasted values

    Returns:
        Float64: MAPE metric

"""
function MAPE(y_test::Vector{Fl}, y_forecast::Vector{Fl})::Float64 where {Fl}
    return mean(abs.((y_test - y_forecast)./y_test))
end

"""
    Calculate the RMSE metric for a given forecast and test set.

    Args:
        y_test::Vector{Fl}: Vector with the test set values
        y_forecast::Vector{Fl}: Vector with the forecasted values

    Returns:
        Float64: RMSE metric

"""
function RMSE(y_test::Vector{Fl}, y_forecast::Vector{Fl})::Float64 where {Fl}
    return sqrt(mean((y_test - y_forecast).^2))
end

"""
    Calculate the nRMSE metric for a given forecast and test set.

    Args:
        y_test::Vector{Fl}: Vector with the test set values
        y_forecast::Vector{Fl}: Vector with the forecasted values

    Returns:
        Float64: nRMSE metric
"""
function nRMSE(y_test::Vector{Fl}, y_forecast::Vector{Fl}) where {Fl}
    return sqrt(mean((y_test - y_forecast).^2))/mean(y_test)
end

"""
    Calculate the MSIS metric for a given forecast, train set, upper and lower bounds.

    Args:
        y_train::Vector{Fl}: Vector with the training set values
        y_test::Vector{Fl}: Vector with the test set values
        U_simulation::Vector{Fl}: Vector with the upper bounds of the simulation
        L_simulation::Vector{Fl}: Vector with the lower bounds of the simulation
        α::Float64: Significance level for the MSIS metric
        s::Int64: Seasonality of the time series

    Returns:
        Float64: MSIS metric
"""
function MSIS(y_train::Vector{Fl}, y_test::Vector{Fl}, U_simulation::Vector{Fl}, L_simulation::Vector{Fl}, s::Int64)::Float64 where {Fl}

    T = length(y_train)
    H = length(y_test)

    L_diff = L_simulation - y_test
    U_diff = y_test - U_simulation

    denominator = (1/(T-s))*sum(abs(y_train[j]-y_train[j-s]) for j in s+1:T)

    return (1/H)*(sum(U_simulation - L_simulation) + (2/α)*sum(L_diff[L_diff .> 0]) + (2/α)*sum(U_diff[U_diff .> 0]))/denominator

end

function CRPS(simulation::Matrix{Float64}, y::Vector{Float64})::Float64
    H, S = size(simulation)
    
    crps_value = 0.0
    for t in 1:H
        
        # Compute the cumulative distribution function (CDF)
        sorted_data = sort(simulation[t, :])
        cdf = [sum(sorted_data .<= x) / length(sorted_data) for x in sorted_data]
        
        # Calculate CRPS
        t_crps_value = 0.0
        for i in 1:S
            if sorted_data[i] < y[t]
                t_crps_value += (1 - cdf[i])^2
            else
                t_crps_value += cdf[i]^2
            end
        end

        crps_value += (t_crps_value / S) / H
    end

    return crps_value
end


"""
    Calculate the COVERAGE metric for a given forecast, test set and quantile.

    Args:
        scenarios::Matrix{T}: Matrix with the scenarios
        y_test::Vector{T}: Vector with the test set values
        q::Float64: Quantile to be calculated

    Returns:
        Float64: COVERAGE metric

"""
function COVERAGE(scenarios::Matrix{Fl}, y_test::Vector{Fl}, q::Float64)::Float64 where {Fl}

    H = size(scenarios, 1)

    quantiles = mapslices(x -> quantile(x, q), scenarios; dims = 2)
    
    return sum(y_test .< quantiles)
end

"""
    Initialize a dictionary to store the metrics for each model, metric and horizon.

    Args:
        model_dict_keys::Vector{String}: Vector with the model names
        n_series::Int64: Number of time series

    Returns:
        Dict{String, Dict{String, Dict{String, Vector{Float64}}}}: Dictionary to store the metrics
"""
function initialize_metrics_dict(model_dict_keys::Vector{String}, n_series::Int64)::Dict{String, Dict{String, Dict{String, Vector{Float64}}}}

    metrics_dict = Dict{String, Dict{String, Dict{String, Vector{Float64}}}}()

    for model_name in model_dict_keys
        metrics_dict[model_name] = Dict{String, Dict{String, Vector{Float64}}}()
        for metric in METRICS 
            metrics_dict[model_name][metric] = Dict{String, Vector{Float64}}()
            for h in HORIZONS
                metrics_dict[model_name][metric][h] = Vector{Float64}(undef, n_series) 
            end
        end
    end

    return metrics_dict
end

"""
    For a new forecast, update the metrics dictionary with the new values.

    Args:
        metrics_dict::Dict{String, Dict{String, Dict{String, Vector{Float64}}}}: Dictionary with the metrics
        prediction::Vector{Float64}: Vector with the forecasted values
        scenarios::Union{Matrix{Float64}, Nothing}: Matrix with the scenarios
        y_train::Vector{Float64}: Vector with the training set values
        y_test::Vector{Float64}: Vector with the test set values
        series_idx::Int64: Index of the time series
        model_name::String: Name of the model
        granularity::String: Granularity of the dataset

    Returns:
        Nothing
"""
function update_metrics!(metrics_dict::Dict{String, Dict{String, Dict{String, Vector{Float64}}}}, y_forecast::Vector{Fl},
                            scenarios::Union{Matrix{Float64}, Nothing}, y_train::Vector{Fl}, y_test::Vector{Fl}, 
                            series_idx::Int64, model_name::String, granularity::String)::Nothing where {Fl}
                            
    for (h, idxs) in WINDOWS_HORIZON_DICT[granularity]
        metrics_dict[model_name]["MASE"][h][series_idx]  = MASE(y_train, y_test[idxs], y_forecast[idxs], GRANULARITY_DICT[granularity]["s"])
        metrics_dict[model_name]["MAPE"][h][series_idx]  = MAPE(y_test[idxs], y_forecast[idxs])
        metrics_dict[model_name]["sMAPE"][h][series_idx] = sMAPE(y_test[idxs], y_forecast[idxs])
        metrics_dict[model_name]["RMSE"][h][series_idx]  = RMSE(y_test[idxs], y_forecast[idxs])
        metrics_dict[model_name]["nRMSE"][h][series_idx] = nRMSE(y_test[idxs], y_forecast[idxs])
        metrics_dict[model_name]["MAE"][h][series_idx]   = MAE(y_test[idxs], y_forecast[idxs])
        metrics_dict[model_name]["MSE"][h][series_idx]   = MSE(y_test[idxs], y_forecast[idxs])
        
        if !isnothing(scenarios)
            metrics_dict[model_name]["MSIS"][h][series_idx] = MSIS(y_train, y_test[idxs], maximum(scenarios, dims=2)[idxs], minimum(scenarios, dims=2)[idxs], GRANULARITY_DICT[granularity]["s"])
            metrics_dict[model_name]["CRPS"][h][series_idx] = CRPS(scenarios[idxs, :], y_test[idxs])
            for q in [0, 0.1, 0.5, 0.9, 1]
                metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h][series_idx] = COVERAGE(scenarios[idxs, :], y_test[idxs], q)
            end
        else
            metrics_dict[model_name]["MSIS"][h][series_idx] = NaN
            metrics_dict[model_name]["CRPS"][h][series_idx] = NaN
            for q in [0, 0.1, 0.5, 0.9, 1]
                metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h][series_idx] = NaN
            end
        end
    end
end

function update_metrics2(y_forecast::Vector{Fl},
                            scenarios::Union{Matrix{Float64}, Nothing}, y_train::Vector{Fl}, y_test::Vector{Fl}, 
                            series_idx::Int64, model_name::String, granularity::String) where {Fl}
                            
    m_dict = Dict()
    m_dict["MASE"] = Dict()
    m_dict["MAPE"] = Dict()
    m_dict["sMAPE"] = Dict()
    m_dict["RMSE"] = Dict()
    m_dict["nRMSE"] = Dict()
    m_dict["MAE"] = Dict()
    m_dict["MSE"] = Dict()
    m_dict["MSIS"] = Dict()
    m_dict["CRPS"] = Dict()
    for q in [0, 0.1, 0.5, 0.9, 1]
        m_dict["COVERAGE_$(Int64(q*100))"] = Dict()
    end
    for (h, idxs) in WINDOWS_HORIZON_DICT[granularity]
        m_dict["MASE"][h] = Dict()
        m_dict["MAPE"][h] = Dict()
        m_dict["sMAPE"][h] = Dict()
        m_dict["RMSE"][h] = Dict()
        m_dict["nRMSE"][h] = Dict()
        m_dict["MAE"][h] = Dict()
        m_dict["MSE"][h] = Dict()
        m_dict["MSIS"][h] = Dict()
        m_dict["CRPS"][h] = Dict()
        for q in [0, 0.1, 0.5, 0.9, 1]
            m_dict["COVERAGE_$(Int64(q*100))"][h] = Dict()
        end
    end
    for (h, idxs) in WINDOWS_HORIZON_DICT[granularity]
        m_dict["MASE"][h][series_idx]  = MASE(y_train, y_test[idxs], y_forecast[idxs], GRANULARITY_DICT[granularity]["s"])
        m_dict["MAPE"][h][series_idx]  = MAPE(y_test[idxs], y_forecast[idxs])
        m_dict["sMAPE"][h][series_idx] = sMAPE(y_test[idxs], y_forecast[idxs])
        m_dict["RMSE"][h][series_idx]  = RMSE(y_test[idxs], y_forecast[idxs])
        m_dict["nRMSE"][h][series_idx] = nRMSE(y_test[idxs], y_forecast[idxs])
        m_dict["MAE"][h][series_idx]   = MAE(y_test[idxs], y_forecast[idxs])
        m_dict["MSE"][h][series_idx]   = MSE(y_test[idxs], y_forecast[idxs])
        
        if !isnothing(scenarios)
            m_dict["MSIS"][h][series_idx] = MSIS(y_train, y_test[idxs], maximum(scenarios, dims=2)[idxs], minimum(scenarios, dims=2)[idxs], GRANULARITY_DICT[granularity]["s"])
            m_dict["CRPS"][h][series_idx] = CRPS(scenarios[idxs, :], y_test[idxs])
            for q in [0, 0.1, 0.5, 0.9, 1]
                m_dict["COVERAGE_$(Int64(q*100))"][h][series_idx] = COVERAGE(scenarios[idxs, :], y_test[idxs], q)
            end
        else
            m_dict["MSIS"][h][series_idx] = NaN
            m_dict["CRPS"][h][series_idx] = NaN
            for q in [0, 0.1, 0.5, 0.9, 1]
                m_dict["COVERAGE_$(Int64(q*100))"][h][series_idx] = NaN
            end
        end
    end
    return m_dict
end

"""
    Add the OWA metric to the dictionary of average metrics.

    Args:
        dict_average_metrics::Dict{String, DataFrame}: Dictionary with the average metrics
        benchmark_name::String: Name of the benchmark model
        model_names::Vector{String}: Vector with the model names

    Returns:
        Nothing
"""
function add_OWA_metric(dict_average_metrics::Dict{String, DataFrame}, benchmark_name::String, model_names::Vector{String})
    
    number_of_models = length(model_names)
    
    matrix_metrics = Matrix{Union{String, Float64}}(undef, number_of_models, 5)
    matrix_metrics[:, 1] .= model_names
    for (m, model_name) in enumerate(model_names)
        
        for h_i in eachindex(HORIZONS)
            MASE_df  = dict_average_metrics["MASE"]
            sMAPE_df = dict_average_metrics["sMAPE"]
            benchmark_MASE  = MASE_df[findfirst(i -> i == benchmark_name, MASE_df[:, 1]), h_i + 1]
            benchmark_sMAPE = sMAPE_df[findfirst(i -> i == benchmark_name, sMAPE_df[:, 1]), h_i + 1]
            model_MASE  = MASE_df[findfirst(i -> i == model_name, MASE_df[:, 1]), h_i + 1]
            model_sMAPE = sMAPE_df[findfirst(i -> i == model_name, sMAPE_df[:, 1]), h_i + 1]
            matrix_metrics[m, h_i + 1] = OWA(model_MASE, benchmark_MASE, model_sMAPE, benchmark_sMAPE)
        end
    end

    owa_metric_dfs = Dict{String, DataFrame}()
    for (m, model_name) in enumerate(model_names)
        owa_metric_dfs[model_name] = DataFrame(matrix_metrics[m, 2:end][:, :]', ["short", "medium", "long", "total"])
    end
    
    return owa_metric_dfs
end

function add_ACD_metric(metrics_dict::Dict, model_names::Vector{String}, granularity::String)

    # Retornar um dicionário de tamanho igual a quantidade de métricas coverage
    # Cada chave aponta pra um DF, com número de linhas igual ao número de modelos e número de colunas igual ao horizonte

    number_of_models = length(model_names)
    
    acd_metric_dict = Dict{String, Matrix}()
    for model_name in model_names
        acd_metric_dict[model_name] = Matrix{Union{String, Float64}}(undef, 5, length(HORIZONS) + 1)
        acd_metric_dict[model_name][:, 1] .= [0, 0.1, 0.5, 0.9, 1]
        for (j, q) in enumerate([0, 0.1, 0.5, 0.9, 1])
            for (i, h_i) in enumerate(HORIZONS)
                number_of_series = length(metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h_i])
                H = length(collect(WINDOWS_HORIZON_DICT[granularity][h_i]))
                acd_metric_dict[model_name][j, i + 1] = abs(q - sum(metrics_dict[model_name]["COVERAGE_$(Int64(q*100))"][h_i]) / (number_of_series * H))
            end
        end
    end

    acd_metric_dfs = Dict{String, DataFrame}()
    for (k, v) in acd_metric_dict
        acd_metric_dfs[k] = DataFrame(v, ["Quantile", "short", "medium", "long", "total"])
    end

    return acd_metric_dfs
end

"""
    Save the metrics in a CSV file for each model and horizon.
    Inside folder Results, each model will contain its own folder with a .csv file for each horizon.

    Args:
        metrics_dict::Dict{String, Dict{String, Dict{String, Vector{Float64}}}}: Dictionary with the metrics
        benchmark_name::String: Name of the benchmark model
        number_of_series::Int64: Number of time series
        granularity::String: Granularity of the dataset

    Returns:
        Nothing

"""
function save_metrics(metrics_dict::Dict{String, Dict{String, Dict{String, Vector{Float64}}}}, benchmark_name::String, number_of_series::Int64, 
                        granularity::String, errors_series_dict::Dict{String, Vector}, set_val::Int64)

    number_of_models     = length(metrics_dict)
    dict_average_metrics = Dict{String, DataFrame}()

    model_names = collect(keys(metrics_dict))
    matrix_metrics = zeros(number_of_series, length(METRICS), length(HORIZONS), number_of_models)
    for (metric_i, metric) in enumerate(METRICS)

        if !(metric in ["COVERAGE_$(Int64(q*100))" for q in [0, 0.1, 0.5, 0.9, 1]])

            matrix_average_metrics = Matrix{Union{String, Float64}}(undef, number_of_models, 5)
            matrix_average_metrics[:, 1] = model_names
            for (horizon_i, horizon) in enumerate(HORIZONS)
                for (model_i, model_name) in enumerate(model_names)
                    matrix_average_metrics[model_i, horizon_i + 1]  = mean(metrics_dict[model_name][metric][horizon])
                    matrix_metrics[:, metric_i, horizon_i, model_i] = metrics_dict[model_name][metric][horizon]
                end
            end
            dict_average_metrics[metric] = DataFrame(matrix_average_metrics, ["model", "short", "medium", "long", "total"])
        end
    end
        
    acd_metric_dfs = add_ACD_metric(metrics_dict, model_names, granularity)
    owa_metric_dfs = add_OWA_metric(dict_average_metrics, benchmark_name, setdiff(model_names, [benchmark_name]))

    for (model_i, model_name) in enumerate(model_names)
        @info "Saving metrics for model: $model_name"

        try
            mkdir("Results Set $set_val/$(granularity)/$(model_name)")
        catch
            @warn "Directory of $(model_name) already exists"
        end

        for (horizon_i, horizon) in enumerate(HORIZONS)
            df_metrics = DataFrame(matrix_metrics[:, :, horizon_i, model_i], METRICS)
            CSV.write("Results Set $set_val/$(granularity)/$(model_name)/$(horizon).csv", df_metrics)
        end

        @info "Saving OWA metric for model: $model_name"
        # CSV.write("Results/$(granularity)/$(model_name)/OWA.csv", dict_average_metrics["OWA"])
        if model_name != "Naive"
            CSV.write("Results Set $set_val/$(granularity)/$(model_name)/OWA.csv", owa_metric_dfs[model_name])
        end

        @info "Saving ACD metric for model: $model_name"
        CSV.write("Results Set $set_val/$(granularity)/$(model_name)/ACD.csv", acd_metric_dfs[model_name])

        @info "Saving series with errors in estimation/forecasting for model: $(model_name)"
        if !isempty(errors_series_dict[model_name])
            CSV.write("Results Set $set_val/$(granularity)/$(model_name)/errors_series.csv", DataFrame(errors_series_dict[model_name][:, :], :auto))
        end

    end
end