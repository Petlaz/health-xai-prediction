# Data Dictionary — Health_XAI_Project

## Overview
This document describes the variables available in `data/raw/heart_data.csv` after preprocessing and feature-name standardisation. Use it as a reference when building models, generating explanations (LIME/SHAP), and drafting project reports.

## Feature Descriptions
| Feature | Description | Type | Example |
|---------|-------------|------|---------|
| cntry | Country code of respondent (ISO-2). | Categorical | AT |
| happy | Self-rated happiness on a 0–10 scale. | Ordinal | 8 |
| sclmeet | Frequency of social meetings with friends, relatives, or colleagues. | Ordinal | 4 |
| inprdsc | How often participates in organised social, religious, or community activities. | Ordinal | 1 |
| health | Self-rated general health (1 very good to 5 very bad). | Ordinal | 3 |
| ctrlife | Feeling of control over life (0 no control to 10 complete control). | Ordinal | 8 |
| etfruit | Frequency of fruit consumption. | Ordinal | 3 |
| eatveg | Frequency of vegetable consumption. | Ordinal | 3 |
| dosprt | Frequency of doing sports or physical exercise. | Ordinal | 3 |
| cgtsmok | Cigarette smoking status/frequency (daily, occasional, former, never). | Categorical | 4 |
| alcfreq | Alcohol drinking frequency. | Ordinal | 3 |
| height | Self-reported height in centimeters. | Continuous | 178 |
| weighta | Self-reported weight in kilograms. | Continuous | 90 |
| fltdpr | How often felt depressed in the last week. | Ordinal | 1 |
| flteeff | How often felt everything was an effort in the last week. | Ordinal | 1 |
| slprl | How often sleep was restless in the last week. | Ordinal | 1 |
| wrhpp | How often felt happy in the last week (reverse coded). | Ordinal | 3 |
| fltlnl | How often felt lonely in the last week. | Ordinal | 1 |
| enjlf | How often enjoyed life in the last week (reverse coded). | Ordinal | 3 |
| fltsd | How often felt sad in the last week. | Ordinal | 1 |
| hltprhc | Doctor diagnosed heart or circulation problems (1 yes, 0 no). | Binary | 0 |
| hltprhb | Doctor diagnosed high blood pressure (1 yes, 0 no). | Binary | 1 |
| hltprdi | Doctor diagnosed diabetes (1 yes, 0 no). | Binary | 0 |
| gndr | Gender (1 male, 2 female). | Categorical | 1 |
| paccnois | Perceived noise problems in the local area (1 yes, 0 no). | Binary | 0 |

