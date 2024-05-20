"""
    Prepare data for training and testing

    Args:
        df_train::DataFrame: DataFrame with the training data
        df_test::DataFrame: DataFrame with the test data

    Returns:
        train_test_dict::Dict: Dictionary with the training and test data
"""
function build_train_test_dict(df_train::DataFrame, df_test::DataFrame)::Dict{Int, Dict{String, Vector{Float64}}}
    train_test_dict = Dict()

    for i in 1:eachindex(df_train[:, 1])
        y_raw = Vector(df_train[i, :])[2:end]
        y_train_raw = y_raw[1:findlast(i->!ismissing(i), y_raw)]
        T = length(y_train_raw)
        y_train = y_train_raw
        y_test  = Vector(df_test[i, :])[2:end]

        train_test_dict[i] = Dict()
        train_test_dict[i]["train"] = Float64.(y_train)
        train_test_dict[i]["test"]  = Float64.(y_test)
    end

    return train_test_dict
end