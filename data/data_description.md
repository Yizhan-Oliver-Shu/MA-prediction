# M&A deals dataset description

The standard data source for M&A academic study is SDC Platinum, to which many university libraries have subscription. As a general rule, we first download as reasonably large a dataset as possible for once from the licensed library computer, and then further process it on PCs. 


## Filters in SDC:

As the database is very comprehensive, we need to find a reasonable set of filters useful for our specific tasks:

- domestic.
- date 1990 - end of 2021.
- Form of the deal: Merger. 
- Target public status: Public. 
- Target primary exchange: NYSE, Nasdaq, NYSE Amex, American.
- Deal Status: Completed, Withdrawn, Pending.

![SDC filters](SDC_filters.jpeg?raw=true)



## Session and Report Files
To replicate the query result (on computers on A floor Firestone), you can directly import the session file `session.ssh` and report file `report.rpt` into SDC Platinum to query the database. (The database is updated on a daily basis.)

## Variables
The report file contains the acronyms of all the variables in our query. For a quick lookup, the file `column_names.csv` contains the full name for those variables. You can find the exact definition of all the variables available in the database in the file `SDC_MA_guide.pdf`.


## Output file of SDC
SDC can only export an `xls` file `df.xls`. We convert it to `csv` in `Excel`, delete the useless first and last line, and save it as `df.csv`, the input to Python for data processing.

## Data Processing
Now we have five notebooks for data processing, the first four on variable transformation, and the last one on applying filters. Specifically they perform the following:

- Notebook 0: basic cleaning

  - load data. change column names.
  - transform date-like columns to `datetime.date` data type. transform float-like columns to float data type.
  - correct some deal considerations manually.
  - fill missing values.
  
- Notebook 1: match with CRSP database

  - match targets and acquirors with CRSP
  - look for delisting information for completed deals.
  
- Notebook 2: date correction

  - correct the last trading day before announcement.
  - correct announcement date.
  - correct resolution date
  - create duration (in trading days)
  
- Notebook 3: create new variables

  - Create `amend` and `choice` binary variables.
  - 
  - 
  - 
  
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


## Final dataset
After performing all the data processings, we save the final dataset as `df_final.csv`
