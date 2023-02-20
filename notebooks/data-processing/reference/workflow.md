# Goal: filter out deals with as accurate information  as possible, build a prediction model and backtest.

What information should be accurate?

- the match of companies. (ticker/cusip from SDC and permno from CRSP should match. we veryfiy it from mkt prices)
- pricing information and date align: at a certain date what information has been available (mkt has priced in)



# workflow
- data comes in
- clean column names
- clean dtypes (float and datetime).
- clean dates:
  - push some dates to next trading day. 
  - dao before da. 
  - `one_day`, `aone_day` (never used)
  - dr = dw / de.
  - Later we will
    - replace `dr` with `delist_date` for completed deals.
    - when computing premiums, cap `dao` at 126 days before `da`.
- clean ticker and cusip.
- manual correction of consideration.
- fill missing `pr_initial` by `pr`.
- delete deals with: no pricing, no consideration
- adjust deal value by CPI.
- create amendment/choice.
- create competition information
- clean payment type
- extract deal term. (check results, any extreme value?)
- create stock indicator.
- look for permno in CRSP.
- look for delisting
- look for market data from CRSP. update `dr` by delisting date before pulling mkt data.
- combine results. update `dr` before.
- process raw market data.
  - add delisting price and return
  - fill missing price and return
  - calculate mktcap
  - calculate adjusted price.
- 

# to-do
- when calculating premium, `dao` not earlier than 126 days before `da`.

# Filters:
  
  
- lack of accurate information:
    - no pricing, no consideration
    - unsuccessful permno match for the tgt, and the acq in a stock deal. 
    - completed:
      - delisting code not due to MA.
    -  delisting return is missing (complex payment)
      - difference between effective date and delisting date is too large (> 40 trading days)
    - dae

- inapplicable to our research:
    - small deals
    - duration too short/long
    - low price ??
  
- too complex for the model to handle:
    - term parsing failed
    -  amendment
    -  competition
    - `dateval` near `da`

# thoughts on cleaning

- no amendment:
  - `valamend`
  - `pr`==`pr_initial`
  - consideration parsed successfully
- `dateval`==`da`. then pricing must be accurate.
- use `dateval` as the starting point for investment.
- 

# to-do
- feature engineering
- process mkt data
