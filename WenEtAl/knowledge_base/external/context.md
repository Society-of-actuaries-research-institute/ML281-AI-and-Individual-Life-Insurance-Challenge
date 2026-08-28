# External Actuarial Context

## Data Source
ILEC (Individual Life Experience Committee) mortality study, observation years 2012–2019.
Data represents US individual life insurance policies from SOA member companies.

## Key Limitations
- COVID-19 pandemic (2020+) is outside training scope; model should not be extrapolated beyond 2019 experience
- Mortality improvement assumptions use SOA improvement scales current as of data collection period
- Product mix in ILEC data may not reflect current market distribution

## Mortality Environment (2012–2019)
- Overall US life expectancy improvement slowed in mid-2010s compared to prior decade
- Opioid epidemic disproportionately affected working-age adults (ages 25–54), particularly in certain geographic regions
- Select-and-ultimate mortality patterns are well-established for individual life; first 15–25 policy years typically show favorable experience relative to ultimate rates

## Actuarial Standards Context
- **ASOP 23**: Data quality — ILEC data is publicly available but company-reported; classification errors possible
- **ASOP 25**: Credibility — segments with exposure < 1,000 life-years should be treated as low-credibility; use Bühlmann credibility blending
- **ASOP 41**: Communication — clearly distinguish model predictions from actual emerging experience
- **ASOP 56**: Modeling — document model limitations, validate against held-out periods, communicate uncertainty

## VBT 2015 Reference
The ILEC dataset includes VBT 2015 expected deaths as a benchmark. The LightGBM model is trained to predict actual/expected relative to company-specific experience, which may differ from VBT 2015 population-average assumptions.
