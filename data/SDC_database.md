# Querying SDC Platinum for M&A Deals Data

We downloaded the M&A deals data from SDC Platinum on a licensed computer in the university library. As the database is very comprehensive, we need to find a reasonable set of filters useful for our specific tasks. 

The general rule is to download as reasonably large a dataset as possible from the library computer, and then further clean it on our own PCs. This will save the time back and forth between library and office.


## Filters in SDC:

![SDC filters](SDC_filters_short.jpeg?raw=true)



- domestic.
- date 1990-2021.
- Form of the deal: Merger. (53666)
- Target public status: Public. (14926)
- Target primary exchange: NYSE, Nasdaq, NYSE Amex, American. (10977)
- Deal Status: Completed, Withdrawn, Pending. (9854)

## Session and report files
You can directly import the session file `session.ssh` and report file `report.rpt` in SDC Platinum to query the database. 

## Variable definition
The report file contains the column names of the dataset which we will extract later in Python. The file `SDC_MA_guide.pdf` provides the exact definition of all the variables available in the database.