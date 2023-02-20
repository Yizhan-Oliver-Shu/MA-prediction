# M&A Deals Dataset Processing

The pipeline for data processing is that: after 

- (data-0: notebook prefix) basic processing, 

we perform the following two in parallel:

- (data-1) advanced processing on the variables.
- match with CRSP database:
  - (data-2) look for permnos,
  - (data-3) pulling delisting information, and market data.
  

Then we 

- (data-4) combine the previous results into one dataset `df_processed`. 
- (data-5) process raw market data.

Next we describe in detail what we do in each of the notebooks.


# data-0: basic processing

- Process column names: replace full names with acronyms.
- Process data types: convert dtypes for float-like and date-like (saved as `datetime.date`) columns. 
- Process dates:
    - some dates must be trading days. 
    - original annnouncement date `dao` to be before announcement date `da`. Later we would adjust it to be within $[\text{da}-126, \text{da}]$ when calculating premiums.
    - create `dr` date of resolution (end of deal), that is effective date `de` for completed deals, withdrawal date `dw` for withdrawn deals and missing for pending deals. Later for completed deals we will replace `de` with `delist_date` found in CRSP.
- Clean tickers and cusips.
- Process deal consideration: some manual correction.
- Fill missing `pr_initial` by `pr`.


Basic filterings: delete the following deals

- not applicable to our research (detected manually).
- Price is missing.
- Deal consideration is missing. 

# data-1: advanced processing

- Adjust deal value by CPI.
- Create `amend` and `choice` binary variables.
- Create competing group number and competing status code.
- Clean payment type.
- Extract cash and stock terms from consideration.
- Create stock deal indicator. 

# data-2: CRSP, look for permno


We do the following for both targets and acquirors.

- first look for `permno` by ticker and `da`.
- pull out the market prices 5 days before and after `dao`, and match with the columns `['pr1day', 'tprday', 'tpr1daya']` in SDC. If they are close, then the match is successful.
- Otherwise look for `permno` by `cusip` and `da`, and then check stock price in the same manner.

# data-3: CRSP, pull delist and market data

- Look for delisting information (delisting code, last trade date, delisting date and delisting returns) in CRSP database, for all the targets.
- pull raw market data for all the targets and acquirors, from 40 (trading) days prior to `dao`, to 40 days after `dr`.



# data-4: combine processing results


- replace `dr` with `delist_date` for certain deals.
- create duration.
- combine the dataset after advanced processing and CRSP query. 




# data-5: process market data



## General guidelines for data cleaning & processing:

- We create new columns (variables)  on all the rows first, before applying  any  filters. 
- When filtering we should not drop any row directly, in case we want to retrieve them later. Instead we add another column called `retain` to indicate whether to retain the row after applying the filters. 
- These notebooks shall be highly modular, meaning that almost every data operation should be encapsulated in a function in the helper package. Each function should be developed in a separate notebook. In this way the end user only needs to read the documentation without digging into the codes.
- From time to time we save the intermediate result as an `hdf` file, as some codes (especially those querying the CRSP database) need tens of minutes to run. Thus we want to run it just for once and store the results for later use. The advantage of `hdf` over `csv` is that it preserves data type like `datetime.date`. Only when we need to inspect the dataset by `Excel` or `Numbers` shall we save it as `csv`.
