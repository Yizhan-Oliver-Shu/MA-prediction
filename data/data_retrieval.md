# M&A deals dataset description

The standard data source for M&As academic study is SDC Platinum, to which many university libraries have subscription. As a general rule, we first download as reasonably large a dataset as possible for once from a licensed library computer, and then further process it on PCs. 


## Filters in SDC:

As the database is very comprehensive, we need to find a reasonable set of filters useful for our specific tasks:

- domestic.
- date: 01/01/1984 - 12/31/2022.
- Form of the deal: Merger. 
- Target public status: Public. 
- Target primary exchange: NYSE, Nasdaq, NYSE Amex, American.
- Deal Status: Completed, Withdrawn, Pending.

![SDC filters](SDC_filters.JPG?raw=true)



## Session and Report Files
To replicate the query result (on computers on A floor Firestone), you can directly import the session file `session.ssh` and report file `report.rpt` into SDC Platinum to query the database. (The database is updated on a daily basis.)

## Variables
The report file contains the acronyms of all the variables in our query. For a quick lookup, the file `column_names.csv` contains the full name for those variables. You can find the exact definition of all the variables available in the database in the file `SDC_MA_guide.pdf`.


## Output file of SDC
SDC can only export an `xls` file `df.xls`. We convert it to `csv` in `Excel`, delete the useless first and last line, and save it as `df.csv`, the input to Python for data processing.

## Final dataset
After performing all the data processing (described in another file), we save the final dataset as `df_final.csv`
