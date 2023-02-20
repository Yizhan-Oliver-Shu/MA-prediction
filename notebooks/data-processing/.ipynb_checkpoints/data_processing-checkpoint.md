# M&A Deals Dataset Processing

The pipeline for data processing is that: after 

- (data-0: notebook prefix) basic processing, 

we perform the following two in parallel:

- (data-1) advanced processing on the variables.
- match with CRSP database:
  - (data-2) look for permnos.
  - (data-3) pulling delisting information, and market data
  - (data-4) process raw market data.

Finally we (data-5) combine the previous results into one dataset `df_processed`.

Next we describe in detail what we do in each of the notebooks.


# data-0: basic processing
- Change column names from full names to acronyms.
- Transform datatypes: float-like and date-like (to `datetime.date`). 
- Correct some dates 
    - some dates must be trading days. 
    - `dao` to be within $[\text{da}-126, \text{da}]$.
    - create `dr` date of resolution, that is `de` for completed deals and `dw` for withdrawn deals. Later for completed deals we will replace `de` with `delist_date` found in CRSP.
- Some manual correction of the dataset, mainly about the deal consideration string.


# data-1: advanced processing

# data-2: merge with CRSP


Look for the `permno`s of all the targets and acquirors in the following way:
- first look for `permno` by ticker and `da`.
- pull out the data 5 days before and after `dao`, and match with the columns `['pr1day', 'tprday', 'tpr1daya']` in SDC. If they are close, then the match is successful.
- Otherwise look for `permno` by `cusip` and `da`, and then check stock price in the same manner.

# data-3: merge with CRSP
# data-4: merge with CRSP
# data-5: merge with CRSP



- Look for delisting information (delisting code, last trade date, delisting date and delisting returns) in CRSP database, for all the targets.
- pull raw market data for all the targets and acquirors, from 40 (trading) days prior to `dao`, to 40 days after `dr`.





## General guidelines for data cleaning & processing:

- We create new columns (variables)  on all the rows first, before applying  any  filters. 
- When filtering we should not drop any row directly, in case we want to retrieve them later. Instead we add another column called `retain` to indicate whether to retain the row after applying the filters. 
- These notebooks shall be highly modular, meaning that almost every data operation should be encapsulated in a function in the helper package. Each function should be developed in a separate notebook. In this way the end user only needs to read the documentation without digging into the codes.
- From time to time we save the intermediate result as an `hdf` file, as some codes (especially those querying the CRSP database) need tens of minutes to run. Thus we want to run it just for once and store the results for later use. The advantage of `hdf` over `csv` is that it preserves data type like `datetime.date`. Only when we need to inspect the dataset by `Excel` or `Numbers` shall we save it as `csv`.




(in order):

- Notebook 0: basic cleaning.
- Notebook 1: match with CRSP database.
- Notebook 2: date correction.
- Notebook 3: pull market data from CRSP.
- Notebook 4: create new variables.
- Notebook 5: process market data.
- Notebook 6: create variables for prediction model.
- Notebook 7: apply filters.






  - Load column acronyms from the report file. Load raw data. Change column names. 
  - Transform date-like columns to `datetime.date` dtype. Transform float-like columns to float.
  - Correct `consid` for some deals manually.
  - Fill missing:
      - `pr_initial` by `pr`. 
      - `one_day` by the previous trading day to `dao`.
  
- Notebook 1: match with CRSP database

  - Look for the `permno` of targets and acquirors in CRSP database.
  - For completed deals, look for delisting code, dates and returns in CRSP database.
  
- Notebook 2: date correction

  - correct the last trading day before announcement `one_day`: correct it to be indeed a trading day.
  - correct announcement date `da`: one of `da` and `da`+1 trading day that has the higher trading volumes.
  - correct resolution date `dr`
  - create deal duration: the number of trading days between `da_corrected` and `dr_corrected`.

- Notebook 3: pull market data
  - pull the market data of targets and acquirors during the whole deal process (from 10 days before announcement date to 10 days after resolution date). 

  
- Notebook 4: create new variables

  - Create `amend` and `choice` binary variables.
  - Create competing group number and competing status code.
  - Create deal value adjusted by CPI index.
  - Create new payment type.
  - Extract cash and stock terms from deal consideration.
  
## Filters
- Notebook 4: apply filters

Apply filters:

- data error detected manually.
- deal price, initial deal price >= \$1. 
- target stock price 1 day, 1 week and 4 weeks before >= \$1.
- deal value adjusted by inflation >= \$200 m.
- date announced is not estimated.
- deal cash and stock terms parsing successful.
- target and acquiror matched in CRSP.
- For completed deals, delisting code in CRSP is M&A related.
- For completed deals, effective date in SDC and target delisting date in CRSP is close. 
- duration no fewer than 5 trading days.


Results of filtering:

- delete 3 deals due to data error. 9854 -> 9851
- delete 1503 deals due to deal price smaller than \$1. 9851 -> 8348
- delete 490 deals due to target price smaller than \$1. 8348 -> 7858
- delete 2473 deals due to deal value adjusted by inflation smaller than $200 m. 7858 -> 5385
- delete 0 deals due to estimated announcement date. 5385 -> 5385
- delete 1117 deals due to failure of extracting cash and stock terms. 5385 -> 4268
- delete 31 deals due to failure of matching target in CRSP. 4268 -> 4237
- delete 33 deals due to failure of matching acquiror in CRSP for stock deals. 4237 -> 4204
- delete 21 deals due to non-M&A-related delisting code for completed deals. 4204 -> 4183
- delete 10 deals due to mismatch between delisting date in CRSP and effective date in SDC. 4183 -> 4173
- delete 39 deals due to duration shorter than 5 trading days. 4173 -> 4134

## New variables created

## Final dataset
After performing all the data cleaning and processing, we save the final dataset as `df_final.csv` in `/data/final`.
