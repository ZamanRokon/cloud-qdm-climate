# Data licensing and governance

The Apache-2.0 license covers only this repository's code and documentation.
It does not relicense source climate data or derived products where provider
terms continue to apply.

## MSWEP

MSWEP is not included. Obtain it directly from
[GloH2O](https://www.gloh2o.org/). GloH2O distributes MSWEP under CC BY-NC 4.0
for eligible noncommercial uses; commercial use requires a separate license.
Each user is responsible for checking the current terms, attribution,
redistribution, and derived-product requirements.

Do not commit MSWEP files, create public download links, or operate a public
service that redistributes MSWEP without explicit authorization.

## CHIRPS

[CHIRPS v2 Daily](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
is accessed from the Earth Engine catalog and is identified there as public
domain. Record the exact collection ID and version and cite the provider's
dataset page and scientific reference.

## ERA5-Land

[ERA5-Land Daily Aggregated](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR)
is provided through Google/Copernicus. Follow the attribution and license
information on the Earth Engine catalog and Copernicus Climate Data Store.
Record the access date.

## NEX-GDDP-CMIP6

The [NEX-GDDP-CMIP6 catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_GDDP-CMIP6)
states that images carry model-specific license metadata. Available model
outputs may be CC BY 4.0 or CC0, and their original CMIP6 terms apply. Preserve
the selected models, scenarios, collection version, and data-access date in
published provenance.

## Personal data and credentials

The workflow does not require personal data. Google tokens and service-account
credentials must remain in the runtime's credential store or Colab Secrets.
They must never be written into YAML, notebook outputs, Git remotes, logs, or
source files.
