# The DA Collective

In this file the submission is described from the DA Collective

## Author(s)

- Jeremy White (Intera)
- Nick Martin (SWRI)
- Rui Hugman (Intera)
- Mike Fienen (USGS)
- Ed de Sousa (Intera)

## Modeled locations

We modelled the following locations (check modelled locations):

- [x] Netherlands
- [x] Germany
- [X] Sweden 1
- [x] Sweden 2
- [x] USA

## Model description

We used the model Pastas as described in detail in Collenteur et al. (2019). This is a TFN type of model. The model is 
implemented in the Pastas software package that was used here.  We used PESTPP-IES and a lot of dirty tricks to derive a highly-subjective posterior ensemble of Pastas parameters and forcing multipliers that were used to produce a best-fit and upper/lower confidence interval.  This is all detailed in the provided workflow

## Model workflow to reproduce

Everything is encapsulated in workflow/ dir and is driven by "workflow.py". Specific version of Pastas and pyEMU as included, as are binaries for PESTPP-IES

## Supplementary model data used

'No additional information was obtained and/or used.'

## Estimation of effort

The development time is spread over all the locations as the scripting was developed to work for all locations.  The calibration time is how long it takes PESTPP-IES to run for each location.  Note there were several iterations of the calibration process - these are baked into the development time estimates

| Location    | Development time (hrs) | Calibration time (hrs) | Total time (hrs) | 
|-------------|------------------------|-----------------------|------------------|
| Netherlands | ~ 4                    |    2                  |  6                |
| Germany     | ~ 4                    |    2                  |  6                |
| Sweden 1    | ~ 4                    |    2                  |  6                |
| Sweden 2    | ~ 4                    |    2                  |  6                |
| USA         | ~ 4                    |    2                  |  6                |

## Additional information

There looks to be about a year of time missing for the "netherlands" location from sept 2015 (end of historic period) to sept 2016 in the example submission file so we re-indexed our results to align with that datetime index.
