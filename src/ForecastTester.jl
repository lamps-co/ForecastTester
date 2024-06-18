module ForecastTester

using CSV, DataFrames, StateSpaceModels, Statistics, PyCall, Distributions, Distributed, RCall



include("../StateSpaceLearning/src/StateSpaceLearning.jl")

include("preparedata.jl")
include("metrics.jl")
include("models/utils.jl")
include("models/Naive.jl")
include("models/StateSpaceModels.jl")
include("models/AutoSarimaPython.jl")
include("models/ETS.jl")
include("models/StateSpaceModelsPython.jl")

const GRANULARITY_DICT = Dict("monthly"   => Dict("s" => 12, "H" => 18),
                              "daily"     => Dict("s" => 1, "H" => 14),
                              "weekly"    => Dict("s" => 1, "H" => 13),
                              "quarterly" => Dict("s" => 1, "H" => 8),
                              "hourly"    => Dict("s" => 24, "H" => 48),
                              "yearly"    => Dict("s" => 1, "H" => 6))

const WINDOWS_HORIZON_DICT = Dict("monthly"   => Dict("short" => 1:6, "medium" => 7:12, "long" => 13:18, "total" => 1:18),
                                  "daily"     => Dict("short" => 1:4, "medium" => 5:9, "long" => 10:14, "total" => 1:14),
                                  "weekly"    => Dict("short" => 1:4, "medium" => 5:9, "long" => 10:13, "total" => 1:13),
                                  "quarterly" => Dict("short" => 1:2, "medium" => 3:5, "long" => 6:8, "total" => 1:8),
                                  "hourly"    => Dict("short" => 1:16, "medium" => 17:32, "long" => 33:48, "total" => 1:48),
                                  "yearly"    => Dict("short" => 1:2, "medium" => 3:4, "long" => 5:6, "total" => 1:6))

const HORIZONS = ["short", "medium", "long", "total"]
const S = 1000
const METRICS = ["MASE", "sMAPE", "MAPE", "RMSE", "nRMSE", "MAE", "MSE", "MSIS", "COVERAGE_0", "COVERAGE_100", "COVERAGE_10", "COVERAGE_90", "COVERAGE_50"]
const α = 0.05
"""
    Read datasets from CSV files of a specified grannularity and return them as DataFrames.

    Args:
        granularity::String: Granularity of the dataset to be read

    Returns:
        monthly_train::DataFrame: DataFrame with the training data
        monthly_test::DataFrame: DataFrame with the test data
"""
function read_dataframes(granularity::String)::Tuple{DataFrame, DataFrame}

    train_set = CSV.read("datasets/$granularity-train.csv", DataFrame)
    test_set  =  CSV.read("datasets/$granularity-test.csv", DataFrame)

    return train_set, test_set
end

function initialize_dict_with_errors_series(model_dict::Dict)::Dict{String, Vector}

    errors_series = Dict{String, Vector}()

    for (model_name, model_function) in model_dict
        errors_series[model_name] = []
    end

    return errors_series
end

function run_distributed(input::Dict)

    y_train    = input["train"]
    i          = input["i"]
    model_dict = input["model_dict"]
    s          = input["s"]
    H          = input["H"]

    (i % 1000) == 1 ? printstyled("Run series $(i)\n"; color = :green) : nothing

    output_dict = Dict()
    errors_series_dict_i = Dict()

    for (model_name, model_function) in model_dict
        
        prediction = nothing; simulation = nothing
        try
            prediction, simulation = model_function(y_train, s, H, ForecastTester.S)
            if i == 2
                throw("erro")
            end
        catch
            printstyled("Error when estimating/forecasting model $(model_name)!\n"; color = :red)
            prediction = ones(H) .* y_train[end]
            simulation = nothing

            errors_series_dict_i[model_name] = i
        end
        output_dict[model_name] = Dict()
        output_dict[model_name]["prediction"] = prediction
        output_dict[model_name]["simulation"] = simulation
    end

    return Dict("output_dict" => output_dict, "i" => i, "errors_series_dict_i" => errors_series_dict_i)

end

"""
    Run the requested models and granularity and save the metrics in a CSV file.
    
    Args:
        test_function::Dict{String, Fn}: Dictionary with the model names and their respective functions
        granularity::String: Granularity of the dataset to be Read

    Returns:
        Nothing
"""
function run(test_function::Dict{String, Fn}, granularity::String)::Nothing where {Fn}

    benchmark_function = Dict("Naive" => ForecastTester.get_forecast_naive)

    s = ForecastTester.GRANULARITY_DICT[granularity]["s"]
    H = ForecastTester.GRANULARITY_DICT[granularity]["H"]

    model_dict = merge(test_function, benchmark_function)
    data_dict  = ForecastTester.build_train_test_dict(ForecastTester.read_dataframes(granularity)...)
    
    metrics_dict = ForecastTester.initialize_metrics_dict(collect(keys(model_dict)), length(data_dict))
    errors_series_dict = ForecastTester.initialize_dict_with_errors_series(model_dict)

    prediction = nothing
    simulation = nothing
    
    vec_dict = []
    for i in sort(collect(keys(data_dict)))
        y_train = data_dict[i]["train"]

        push!(vec_dict, Dict("train" => y_train, "i" => i, "model_dict" => model_dict, "s" => s, "H" => H))
    end
    
    output_vec_dict = pmap(run_distributed, vec_dict)

    for j in eachindex(output_vec_dict)

        output_i             = output_vec_dict[j]
        
        i                    = output_i["i"]
        errors_series_dict_i = output_i["errors_series_dict_i"]
            
        y_train = data_dict[i]["train"]
        y_test  = data_dict[i]["test"]
        for model_name in keys(model_dict)
        
            prediction           = output_i["output_dict"][model_name]["prediction"]
            simulation           = output_i["output_dict"][model_name]["simulation"] 

            ForecastTester.update_metrics!(metrics_dict, prediction, simulation, y_train, y_test, i, model_name, granularity)
            
            if haskey(errors_series_dict_i, model_name)
                push!(errors_series_dict[model_name], i)
            end

        end
        
    end

    try
        mkdir("Results")
    catch
        @warn "Directory of Results already exists"
    end
    
    try 
        mkdir("Results/$(granularity)")
    catch
        @warn "Directory of $(granularity) already exists"
    end
    
    save_metrics(metrics_dict, collect(keys(benchmark_function))[1], length(data_dict), granularity, errors_series_dict)
end

end 
