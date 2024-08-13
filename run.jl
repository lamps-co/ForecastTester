import Pkg
Pkg.activate(".")
Pkg.instantiate()

using RCall

R"""
library(forecast)
library(stats)
"""

using ForecastTester 

using Distributed 
addprocs(2)

@everywhere begin 
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    using ForecastTester 

    include("src/ForecastTester.jl")

    test_function = Dict("State Space Python" => ForecastTester.get_forecast_SSM)

    granularity = "monthly"
end

output = ForecastTester.run(test_function, granularity; number_of_series = 10, save_intermediate_results = false)

# -------------------------------------------------------------------------------------------------------------------------- #

using Plots, CSV, DataFrames

plots_idx = [1, 47, 263, 370, 396, 477, 584, 659, 675, 706, 760, 767, 781, 783, 796, 804, 835, 905, 908, 978]
model_names = collect(keys(test_function))

try
    mkdir("Plots")
catch
    @warn("A folder named Plots already exists.")
end

for m_name in model_names
    try
        mkdir("Plots/$(m_name)")
    catch
        @warn("A folder named Plots/$(m_name)")
    end
end

series_idx = CSV.read("runned_series.csv", DataFrame)[:, 1]

for (i, idx) in enumerate(plots_idx)
    for m_name in model_names
        y_train = output[i]["output_dict"][m_name]["y_train"]
        y_test  = output[i]["output_dict"][m_name]["y_test"]
        T = length(y_train)
        H = length(y_test)
        forec      = output[i]["output_dict"][m_name]["prediction"]
        simulation = output[i]["output_dict"][m_name]["simulation"] 

        maximum_scenario = maximum(simulation, dims = 2)
        minimum_scenario = minimum(simulation, dims = 2)

        plt = plot(collect(1:T + H), vcat(y_train, y_test), label = "Historic", color = :black, linewidth = 2)
        plot!(plt, collect(T + 1:T + H), forec, ribbon = (forec .- minimum_scenario, maximum_scenario .- forec),
                    title = "Series $(series_idx[i])", label = "Forecast", color = :green, linewidth = 2)
        savefig(plt, "Plots/$(m_name)/series_$(series_idx[i]).png")
    end
end

