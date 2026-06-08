# Examples

## Benchmark/Index returns

### simple
calculate benchmark returns on trading days using the positions and stock_returns

### help
```python

/execute
bm_pos = position_history.benchmark_weight.unstack(level='instrument_id').reindex(stock_returns.index).ffill()
idx_pos = position_history.index_weight.unstack(level='instrument_id').reindex(stock_returns.index).ffill()

ts = pd.DataFrame()
ts["benchmark"] = (stock_returns * bm_pos).sum(axis=1)
#ts["index"] = (stock_returns * idx_pos).sum(axis=1)
#ts["active"] = ts["index"] - ts["benchmark"]

# Final result
result = (ts+1).cumprod() * 100
```

### 
the above is a good starting point, but it basically means that we are switching to the effective_date weights every day.
This is not what I want. I want now to roll the benchmark and index allocation in $ terms for each stock in line with the performance of the stock
the opening $ of the stock * return gives the closing $ for that stock which is then used to calc the $ NAV of index and bm as well as to normalise the weights for the next day


###
bm_pos = position_history.benchmark_weight.unstack(level='instrument_id').reindex(stock_returns.index).ffill()
ts = (stock_returns * bm_pos).sum(axis=1)
w = bm_pos.sum(axis=1)

# Final result
result = ts

(ts+1).cumprod()*100





## Factor Exposures
calculate the factor exposure of the benchmark and index at the last 




/execute





# Check if index_weight and benchmark_weight sums are approximately equal to 1 across all stocks for all dates
result = (position_history.groupby("effective_date").sum() - 1).abs().sum()

result = weights_non_negative and weights_sum_approx_one


# bm returns

## 
/execute
bm_pos = position_history.benchmark_weight.unstack(level='instrument_id').reindex(stock_returns.index).ffill()
ts = (stock_returns * bm_pos).sum(axis=1)
cumts = (ts+1).cumprod() * 100
result = cumts.iloc[-10:]


# stock vol

/execute result = stock_returns.std().head(10)*np.sqrt(252)

/execute result = stock_returns.std().mean()*np.sqrt(252)

/execute result = stock_returns.rolling(50).std().aggregate(["min", "max"]).mean(axis=1)*np.sqrt(252)



# Vol calc
calculate the average stock volatility (annualised)

calculate rolling 60 day volatility for each stock and pick the max and min volatility and then average across all stocks

calculate rolling 60 day volatility annualised for each stock and pick the max and min volatility and their ratio (max/min) and then average across all stocks

calculate the ratio of the max / min rolling 60 day volatility for each stock

Now create random normal simulations of the same size with unit standard deviation; calculate the rolling 60 days volatility and the ratio of max/min of that volatility for each stock


# portfolio return
calculate the benchmark value on a daily trading basis by
* investing 100 into the benchmark positions on the first effective date, 
* evolving the positions between the rebalance effective_dates as per stock returns 
* and aggregating the benchmark value at every trading day


/execute
simulations = np.random.normal(loc=0, scale=1, size=position_history.shape)
rolling_volatility_sim = pd.DataFrame(simulations, index=position_history.index, columns=position_history.columns).rolling(window=60).std()
max_volatility_sim = rolling_volatility_sim.max()
min_volatility_sim = rolling_volatility_sim.min()
volatility_ratio_sim = max_volatility_sim / min_volatility_sim
descriptive_stats_sim = volatility_ratio_sim.describe()
result = descriptive_stats_sim

/execute
simulations = np.random.normal(loc=0, scale=1, size=position_history.shape)
simulations = pd.DataFrame(simulations, index=position_history.index, columns=position_history.columns)
result = simulations.shape
rolling_volatility_sim = simulations.rolling(window=60).std()
max_volatility_sim = rolling_volatility_sim.max()
min_volatility_sim = rolling_volatility_sim.min()
volatility_ratio_sim = max_volatility_sim / min_volatility_sim
descriptive_stats_sim = volatility_ratio_sim.describe()
result = descriptive_stats_sim

/execute
# Calculate rolling 60-day volatility
rolling_vol = stock_returns.rolling(window=60).std() * (252 ** 0.5)

# Get max and min volatility for each stock
max_vol = rolling_vol.max()
min_vol = rolling_vol.min()

# Calculate the ratio of max to min for each stock
volatility_ratio = max_vol / min_vol

# Get descriptive statistics on the volatility ratios
result = volatility_ratio.describe()



lsof -ti :8000 | xargs -r kill -9


