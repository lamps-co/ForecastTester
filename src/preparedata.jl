#TODO Check if it works with other granularities
function build_train_test_dict(df_train::DataFrame, df_test::DataFrame)
    train_test_dict = Dict()
    for i in eachindex(df_train[:, 1])
        y_raw = Vector(df_train[i, :])[2:end]
        y_train_raw = y_raw[1:findlast(i->!ismissing(i), y_raw)]
        T = length(y_train_raw)
        y_train = y_train_raw
        y_test  = Vector(df_test[i, :])[2:end]

        train_test_dict[i] = Dict()
        train_test_dict[i]["train"] = y_train
        train_test_dict[i]["test"]  = y_test
    end
    return train_test_dict
end