R"library(stats)"
R"library(forecast)"

"""
    Get the forecast using the seasonal naive method.

    Args:
        y::Vector{Float64}: Time series data
        s::Int64: Seasonality
        H::Int64: Forecast horizon
        S::Int64: Number of scenarios for simulation

    Returns:
        prediction::Vector{Float64}: Forecasted values
        scenarios::Nothing
"""
function get_forecast_naive(y::Vector{Float64}, s::Int64, H::Int64, S::Int64)::Tuple{Vector{Float64}, Nothing}

    @rput y s H

    R"""
    SeasonalityTest <- function(input, ppy){
    #Used to determine whether a time series is seasonal
    tcrit <- 1.645
    if (length(input)<3*ppy){
        test_seasonal <- FALSE
    }else{
        xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
        clim <- tcrit/sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
        test_seasonal <- ( abs(xacf[ppy]) > clim[ppy] )
        
        if (is.na(test_seasonal)==TRUE){ test_seasonal <- FALSE }
    }
    
    return(test_seasonal)
    }
    input <- ts(y, frequency = s)
    ST <- FALSE
    if (s > 1){ST <- SeasonalityTest(input,s) }
    if (ST == TRUE){
        dec <- decompose(input, type= "multiplicative")
        des_input <- input/dec$seasonal
        SIout <- head(rep(dec$seasonal[(length(dec$seasonal)-s+1):length(dec$seasonal)], H), H)
    }else{
        des_input <- input ; SIout <- rep(1, H)
    }
    forecast <- naive(des_input, h = H)$mean*SIout
    """

    prediction = @rget forecast
    return prediction, nothing
end

