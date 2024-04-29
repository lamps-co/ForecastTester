module ForecastTester

using CSV, DataFrames, StateSpaceModels, Statistics

include("preparedata.jl")
include("metrics.jl")
include("models/Naive.jl")
include("models/StateSpaceModels.jl")

const GRANULARITY_DICT = Dict("monthly" => Dict("s" => 12, "H" => 18))
const S = 1000
const METRICS = ["MASE", "sMAPE"]

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

function run(test_function::Dict{String, Function}, benchmark_function::Dict{String, Function}, granularity::String)::Nothing

    s = GRANULARITY_DICT[granularity]["s"]
    H = GRANULARITY_DICT[granularity]["H"]

    model_dict = merge(test_function, benchmark_function)
    data_dict  = build_train_test_dict(read_dataframes(granularity)...)
    
    metrics_dict = initialize_metrics_dict(collect(keys(model_dict)), length(data_dict))
    
    for i in keys(data_dict)
        printstyled("Forecasting time-series $i: \n"; color = :yellow)
        y_train = data_dict[i]["train"]
        y_test = data_dict[i]["test"]

        for (model_name, model_function) in model_dict
            printstyled("Model: $(model_name)\n"; color = :green)
            prediction, simulation = model_function(y_train, s, H, S)
            update_metrics!(metrics_dict, prediction, simulation, y_train, y_test, i, model_name)
        end
    end
    add_OWA_metric!(metrics_dict, first(collect(keys(benchmark_function))), collect(keys(test_function)))
    # Calculate OWA
end

end # module ForecastTester
