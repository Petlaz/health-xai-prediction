# ESS Dataset - Feature Definitions and Meanings

## Overview
This document provides comprehensive definitions for all features in the European Social Survey (ESS) health dataset used for health status prediction.

## Target Variable
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| health | Self-reported Health Status | Overall subjective health assessment by respondent | 1=Very Good, 2=Good, 3=Fair, 4=Bad, 5=Very Bad | Ordinal (5-point scale) |

## Health-Related Features

### Direct Health Measures
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| hltprhc | Health Problems - Chronic | Whether respondent has chronic health problems | 0=No, 1=Yes | Binary |
| hltprhb | Health Problems - General | Whether respondent has any health problems | 0=No, 1=Yes | Binary |
| hltprdi | Health Problems - Disability | Whether health problems cause disability | 0=No, 1=Yes | Binary |

### Physical Measurements
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| height | Height | Respondent's height in centimeters | Continuous (cm) | Numerical |
| weighta | Weight | Respondent's weight in kilograms | Continuous (kg) | Numerical |

## Psychological Well-being Features
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| happy | Happiness Level | How happy the respondent is overall | 0-10 scale (0=Extremely unhappy, 10=Extremely happy) | Ordinal |
| fltdpr | Feel Depressed | How often respondent feels depressed | 1=None/Almost none, 2=Some, 3=Most, 4=All/Almost all of the time | Ordinal |
| flteeff | Feel Effective | How often respondent feels effective/capable | 1=None/Almost none, 2=Some, 3=Most, 4=All/Almost all of the time | Ordinal |
| slprl | Sleep Restless | How often respondent feels restless during sleep | 1=None/Almost none, 2=Some, 3=Most, 4=All/Almost all of the time | Ordinal |
| wrhpp | Worth/Happy | How often respondent feels worthwhile/happy | 1=None/Almost none, 2=Some, 3=Most, 4=All/Almost all of the time | Ordinal |
| fltlnl | Feel Lonely | How often respondent feels lonely | 1=None/Almost none, 2=Some, 3=Most, 4=All/Almost all of the time | Ordinal |
| enjlf | Enjoy Life | How much respondent enjoys life | 1=None/Almost none, 2=Some, 3=Most, 4=All/Almost all of the time | Ordinal |
| fltsd | Feel Sad | How often respondent feels sad | 1=None/Almost none, 2=Some, 3=Most, 4=All/Almost all of the time | Ordinal |
| ctrlife | Control Over Life | How much control respondent feels over their life | 0-10 scale (0=No control, 10=Complete control) | Ordinal |

## Social and Lifestyle Features

### Social Interaction
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| sclmeet | Social Meetings | How often respondent meets socially with friends/relatives | 1=Never, 2=Less than once a month, 3=Once a month, 4=Several times a month, 5=Once a week, 6=Several times a week, 7=Every day | Ordinal |
| inprdsc | Personal Development | How much time spent on personal development/learning | 0-6 scale (0=No time, 6=A great deal of time) | Ordinal |

### Health Behaviors
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| etfruit | Eat Fruit | How often respondent eats fruit | 1=Never, 2=Less than once a week, 3=Once a week, 4=2-4 days a week, 5=5-6 days a week, 6=Every day, 7=More than once a day | Ordinal |
| eatveg | Eat Vegetables | How often respondent eats vegetables | 1=Never, 2=Less than once a week, 3=Once a week, 4=2-4 days a week, 5=5-6 days a week, 6=Every day, 7=More than once a day | Ordinal |
| dosprt | Do Sport | How often respondent does sports/physical exercise | 1=Never, 2=Less than once a month, 3=Once a month, 4=Several times a month, 5=Once a week, 6=Several times a week, 7=Every day | Ordinal |
| cgtsmok | Cigarette Smoking | Respondent's cigarette smoking behavior | 1=Never smoked, 2=Used to smoke, 3=Smoke occasionally, 4=Smoke daily, 5=No answer | Ordinal |
| alcfreq | Alcohol Frequency | How often respondent drinks alcohol | 1=Never, 2=Once a month or less, 3=2-4 times a month, 4=2-3 times a week, 5=4+ times a week, 6=Every day | Ordinal |

## Demographic Features
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| gndr | Gender | Respondent's gender | 1=Male, 2=Female | Binary |
| cntry | Country | Country where interview was conducted | 28 European countries (abbreviated codes) | Categorical |
| paccnois | Physical Activity/Noise | Physical activity or noise-related variable | 0=No, 1=Yes | Binary |

## Technical Features
| Feature | Full Name | Description | Values | Type |
|---------|-----------|-------------|---------|------|
| Unnamed: 0 | Row Index | Dataset row identifier | Sequential integers | Identifier |

## Notes
- **Scale Interpretations**: Higher values typically indicate more of the measured attribute
- **Missing Values**: Most features have <1% missing data, indicating high data quality
- **Data Source**: European Social Survey (ESS) - a cross-national survey covering 28 European countries
- **Survey Focus**: Health, well-being, and lifestyle factors affecting quality of life

## Feature Categories for Modeling
1. **Target Variable**: health (5-class ordinal)
2. **Health Indicators**: hltprhc, hltprhb, hltprdi
3. **Physical Measures**: height, weighta (to be converted to BMI)
4. **Psychological Factors**: happy, fltdpr, flteeff, slprl, wrhpp, fltlnl, enjlf, fltsd, ctrlife
5. **Lifestyle Behaviors**: etfruit, eatveg, dosprt, cgtsmok, alcfreq, sclmeet, inprdsc
6. **Demographics**: gndr, cntry (to be dropped)

---
*Last Updated: January 1, 2026*
*Source: European Social Survey Health Module*