# Data Dictionary — Health_XAI_Project

## Overview

This document describes the variables available in `data/raw/heart_data.csv` after preprocessing and feature-name standardisation. Use it as a reference when building models, generating explanations (LIME/SHAP), and drafting project reports.

## Index of Features

[categorical__cntry_AT](#categorical__cntry_at) · [categorical__cntry_BE](#categorical__cntry_be) · [categorical__cntry_BG](#categorical__cntry_bg) · [categorical__cntry_CH](#categorical__cntry_ch) · [categorical__cntry_CY](#categorical__cntry_cy) · [categorical__cntry_DE](#categorical__cntry_de) · [categorical__cntry_ES](#categorical__cntry_es) · [categorical__cntry_FI](#categorical__cntry_fi) · [categorical__cntry_FR](#categorical__cntry_fr) · [categorical__cntry_GB](#categorical__cntry_gb) · [categorical__cntry_GR](#categorical__cntry_gr) · [categorical__cntry_HR](#categorical__cntry_hr) · [categorical__cntry_HU](#categorical__cntry_hu) · [categorical__cntry_IE](#categorical__cntry_ie) · [categorical__cntry_IL](#categorical__cntry_il) · [categorical__cntry_IS](#categorical__cntry_is) · [categorical__cntry_IT](#categorical__cntry_it) · [categorical__cntry_LT](#categorical__cntry_lt) · [categorical__cntry_LV](#categorical__cntry_lv) · [categorical__cntry_ME](#categorical__cntry_me) · [categorical__cntry_NL](#categorical__cntry_nl) · [categorical__cntry_NO](#categorical__cntry_no) · [categorical__cntry_PL](#categorical__cntry_pl) · [categorical__cntry_PT](#categorical__cntry_pt) · [categorical__cntry_RS](#categorical__cntry_rs) · [categorical__cntry_SE](#categorical__cntry_se) · [categorical__cntry_SI](#categorical__cntry_si) · [categorical__cntry_SK](#categorical__cntry_sk) · [hltprhc](#hltprhc) · [numeric__alcfreq](#numeric__alcfreq) · [numeric__cgtsmok](#numeric__cgtsmok) · [numeric__ctrlife](#numeric__ctrlife) · [numeric__dosprt](#numeric__dosprt) · [numeric__eatveg](#numeric__eatveg) · [numeric__enjlf](#numeric__enjlf) · [numeric__etfruit](#numeric__etfruit) · [numeric__fltdpr](#numeric__fltdpr) · [numeric__flteeff](#numeric__flteeff) · [numeric__fltlnl](#numeric__fltlnl) · [numeric__fltsd](#numeric__fltsd) · [numeric__gndr](#numeric__gndr) · [numeric__happy](#numeric__happy) · [numeric__health](#numeric__health) · [numeric__height](#numeric__height) · [numeric__hltprdi](#numeric__hltprdi) · [numeric__hltprhb](#numeric__hltprhb) · [numeric__inprdsc](#numeric__inprdsc) · [numeric__paccnois](#numeric__paccnois) · [numeric__sclmeet](#numeric__sclmeet) · [numeric__slprl](#numeric__slprl) · [numeric__weighta](#numeric__weighta) · [numeric__wrhpp](#numeric__wrhpp)

## Feature Table

| Feature | Description | Type | Example |
|---------|-------------|------|---------|
| categorical__cntry_AT | Country of respondent is Austria (one-hot encoded: AT). | Binary | 1 |
| categorical__cntry_BE | Country of respondent is Belgium (one-hot encoded: BE). | Binary | 0 |
| categorical__cntry_BG | Country of respondent is Bulgaria (one-hot encoded: BG). | Binary | 0 |
| categorical__cntry_CH | Country of respondent is Switzerland (one-hot encoded: CH). | Binary | 0 |
| categorical__cntry_CY | Country of respondent is Cyprus (one-hot encoded: CY). | Binary | 0 |
| categorical__cntry_DE | Country of respondent is Germany (one-hot encoded: DE). | Binary | 0 |
| categorical__cntry_ES | Country of respondent is Spain (one-hot encoded: ES). | Binary | 0 |
| categorical__cntry_FI | Country of respondent is Finland (one-hot encoded: FI). | Binary | 0 |
| categorical__cntry_FR | Country of respondent is France (one-hot encoded: FR). | Binary | 0 |
| categorical__cntry_GB | Country of respondent is United Kingdom (one-hot encoded: GB). | Binary | 0 |
| categorical__cntry_GR | Country of respondent is Greece (one-hot encoded: GR). | Binary | 0 |
| categorical__cntry_HR | Country of respondent is Croatia (one-hot encoded: HR). | Binary | 0 |
| categorical__cntry_HU | Country of respondent is Hungary (one-hot encoded: HU). | Binary | 0 |
| categorical__cntry_IE | Country of respondent is Ireland (one-hot encoded: IE). | Binary | 0 |
| categorical__cntry_IL | Country of respondent is Israel (one-hot encoded: IL). | Binary | 0 |
| categorical__cntry_IS | Country of respondent is Iceland (one-hot encoded: IS). | Binary | 0 |
| categorical__cntry_IT | Country of respondent is Italy (one-hot encoded: IT). | Binary | 0 |
| categorical__cntry_LT | Country of respondent is Lithuania (one-hot encoded: LT). | Binary | 0 |
| categorical__cntry_LV | Country of respondent is Latvia (one-hot encoded: LV). | Binary | 0 |
| categorical__cntry_ME | Country of respondent is Montenegro (one-hot encoded: ME). | Binary | 0 |
| categorical__cntry_NL | Country of respondent is Netherlands (one-hot encoded: NL). | Binary | 0 |
| categorical__cntry_NO | Country of respondent is Norway (one-hot encoded: NO). | Binary | 0 |
| categorical__cntry_PL | Country of respondent is Poland (one-hot encoded: PL). | Binary | 0 |
| categorical__cntry_PT | Country of respondent is Portugal (one-hot encoded: PT). | Binary | 0 |
| categorical__cntry_RS | Country of respondent is Serbia (one-hot encoded: RS). | Binary | 0 |
| categorical__cntry_SE | Country of respondent is Sweden (one-hot encoded: SE). | Binary | 0 |
| categorical__cntry_SI | Country of respondent is Slovenia (one-hot encoded: SI). | Binary | 0 |
| categorical__cntry_SK | Country of respondent is Slovakia (one-hot encoded: SK). | Binary | 0 |
| hltprhc | Doctor diagnosed heart or circulation problems (1 yes, 0 no). | Binary | 0 |
| numeric__alcfreq | Alcohol drinking frequency (scaled numeric feature). | Ordinal | -0.795 |
| numeric__cgtsmok | Cigarette smoking status/frequency (daily, occasional, former, never) (scaled numeric feature). | Ordinal | -0.19 |
| numeric__ctrlife | Feeling of control over life (0 no control to 10 complete control) (scaled numeric feature). | Continuous | 0.31 |
| numeric__dosprt | Frequency of doing sports or physical exercise (scaled numeric feature). | Ordinal | -0.061 |
| numeric__eatveg | Frequency of vegetable consumption (scaled numeric feature). | Ordinal | -0.142 |
| numeric__enjlf | How often enjoyed life in the last week (reverse coded) (scaled numeric feature). | Ordinal | 0.164 |
| numeric__etfruit | Frequency of fruit consumption (scaled numeric feature). | Ordinal | -0.172 |
| numeric__fltdpr | How often felt depressed in the last week (scaled numeric feature). | Ordinal | -0.661 |
| numeric__flteeff | How often felt everything was an effort in the last week (scaled numeric feature). | Ordinal | -0.863 |
| numeric__fltlnl | How often felt lonely in the last week (scaled numeric feature). | Ordinal | -0.61 |
| numeric__fltsd | How often felt sad in the last week (scaled numeric feature). | Ordinal | -0.842 |
| numeric__gndr | Gender (1 male, 2 female) (scaled numeric feature). | Binary | -1.053 |
| numeric__happy | Self-rated happiness on a 0–10 scale (scaled numeric feature). | Continuous | 0.332 |
| numeric__health | Self-rated general health (1 very good to 5 very bad) (scaled numeric feature). | Ordinal | 0.941 |
| numeric__height | Self-reported height in centimeters (scaled numeric feature). | Continuous | 0.731 |
| numeric__hltprdi | Doctor diagnosed diabetes (1 yes, 0 no) (scaled numeric feature). | Binary | -0.265 |
| numeric__hltprhb | Doctor diagnosed high blood pressure (1 yes, 0 no) (scaled numeric feature). | Binary | 1.92 |
| numeric__inprdsc | How often participates in organised social, religious, or community activities (scaled numeric feature). | Ordinal | -1.208 |
| numeric__paccnois | Perceived noise problems in the local area (1 yes, 0 no) (scaled numeric feature). | Binary | -0.221 |
| numeric__sclmeet | Frequency of social meetings with friends, relatives, or colleagues (scaled numeric feature). | Ordinal | -0.492 |
| numeric__slprl | How often sleep was restless in the last week (scaled numeric feature). | Ordinal | -0.937 |
| numeric__weighta | Self-reported weight in kilograms (scaled numeric feature). | Continuous | 0.942 |
| numeric__wrhpp | How often felt happy in the last week (reverse coded) (scaled numeric feature). | Ordinal | 0.141 |

## Detailed Feature Notes

### categorical__cntry_AT

- **categorical__cntry_AT** — Country of respondent is Austria (one-hot encoded).
- **Type:** Binary
- **Example:** 1
- **Values:** {0: Not AT; 1: AT}

### categorical__cntry_BE

- **categorical__cntry_BE** — Country of respondent is Belgium (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not BE; 1: BE}

### categorical__cntry_BG

- **categorical__cntry_BG** — Country of respondent is Bulgaria (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not BG; 1: BG}

### categorical__cntry_CH

- **categorical__cntry_CH** — Country of respondent is Switzerland (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not CH; 1: CH}

### categorical__cntry_CY

- **categorical__cntry_CY** — Country of respondent is Cyprus (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not CY; 1: CY}

### categorical__cntry_DE

- **categorical__cntry_DE** — Country of respondent is Germany (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not DE; 1: DE}

### categorical__cntry_ES

- **categorical__cntry_ES** — Country of respondent is Spain (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not ES; 1: ES}

### categorical__cntry_FI

- **categorical__cntry_FI** — Country of respondent is Finland (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not FI; 1: FI}

### categorical__cntry_FR

- **categorical__cntry_FR** — Country of respondent is France (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not FR; 1: FR}

### categorical__cntry_GB

- **categorical__cntry_GB** — Country of respondent is United Kingdom (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not GB; 1: GB}

### categorical__cntry_GR

- **categorical__cntry_GR** — Country of respondent is Greece (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not GR; 1: GR}

### categorical__cntry_HR

- **categorical__cntry_HR** — Country of respondent is Croatia (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not HR; 1: HR}

### categorical__cntry_HU

- **categorical__cntry_HU** — Country of respondent is Hungary (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not HU; 1: HU}

### categorical__cntry_IE

- **categorical__cntry_IE** — Country of respondent is Ireland (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IE; 1: IE}

### categorical__cntry_IL

- **categorical__cntry_IL** — Country of respondent is Israel (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IL; 1: IL}

### categorical__cntry_IS

- **categorical__cntry_IS** — Country of respondent is Iceland (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IS; 1: IS}

### categorical__cntry_IT

- **categorical__cntry_IT** — Country of respondent is Italy (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IT; 1: IT}

### categorical__cntry_LT

- **categorical__cntry_LT** — Country of respondent is Lithuania (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not LT; 1: LT}

### categorical__cntry_LV

- **categorical__cntry_LV** — Country of respondent is Latvia (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not LV; 1: LV}

### categorical__cntry_ME

- **categorical__cntry_ME** — Country of respondent is Montenegro (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not ME; 1: ME}

### categorical__cntry_NL

- **categorical__cntry_NL** — Country of respondent is Netherlands (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not NL; 1: NL}

### categorical__cntry_NO

- **categorical__cntry_NO** — Country of respondent is Norway (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not NO; 1: NO}

### categorical__cntry_PL

- **categorical__cntry_PL** — Country of respondent is Poland (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not PL; 1: PL}

### categorical__cntry_PT

- **categorical__cntry_PT** — Country of respondent is Portugal (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not PT; 1: PT}

### categorical__cntry_RS

- **categorical__cntry_RS** — Country of respondent is Serbia (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not RS; 1: RS}

### categorical__cntry_SE

- **categorical__cntry_SE** — Country of respondent is Sweden (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not SE; 1: SE}

### categorical__cntry_SI

- **categorical__cntry_SI** — Country of respondent is Slovenia (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not SI; 1: SI}

### categorical__cntry_SK

- **categorical__cntry_SK** — Country of respondent is Slovakia (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not SK; 1: SK}

### hltprhc

- **hltprhc** — Doctor diagnosed heart or circulation problems (1 yes, 0 no).
- **Type:** Binary
- **Example:** 0
- **Values:** {1: yes; 0: no}

### numeric__alcfreq

- **numeric__alcfreq** — Alcohol drinking frequency (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.795
- **Values:** {-0.306, -0.795, -1.284, -1.773, 0.183, 0.673, 1.162}

### numeric__cgtsmok

- **numeric__cgtsmok** — Cigarette smoking status/frequency (daily, occasional, former, never) (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.19
- **Values:** {-0.19, -0.737, -1.284, -1.831, 0.357, 0.904}

### numeric__ctrlife

- **numeric__ctrlife** — Feeling of control over life (0 no control to 10 complete control) (scaled numeric feature).
- **Type:** Continuous
- **Example:** 0.31

### numeric__dosprt

- **numeric__dosprt** — Frequency of doing sports or physical exercise (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.061
- **Values:** {-0.061, -0.447, -0.834, -1.22, 0.325, 0.712, 1.098, 1.485}

### numeric__eatveg

- **numeric__eatveg** — Frequency of vegetable consumption (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.142
- **Values:** {-0.142, -1.041, -1.941, 0.758, 1.657, 2.557, 3.457}

### numeric__enjlf

- **numeric__enjlf** — How often enjoyed life in the last week (reverse coded) (scaled numeric feature).
- **Type:** Ordinal
- **Example:** 0.164
- **Values:** {reverse: coded}

### numeric__etfruit

- **numeric__etfruit** — Frequency of fruit consumption (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.172
- **Values:** {-0.172, -0.937, -1.702, 0.593, 1.358, 2.123, 2.888}

### numeric__fltdpr

- **numeric__fltdpr** — How often felt depressed in the last week (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.661
- **Values:** {-0.661, 0.843, 2.346, 3.85}

### numeric__flteeff

- **numeric__flteeff** — How often felt everything was an effort in the last week (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.863
- **Values:** {-0.863, 0.404, 1.67, 2.937}

### numeric__fltlnl

- **numeric__fltlnl** — How often felt lonely in the last week (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.61
- **Values:** {-0.61, 0.788, 2.186, 3.584}

### numeric__fltsd

- **numeric__fltsd** — How often felt sad in the last week (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.842
- **Values:** {-0.842, 0.634, 2.11, 3.586}

### numeric__gndr

- **numeric__gndr** — Gender (1 male, 2 female) (scaled numeric feature).
- **Type:** Binary
- **Example:** -1.053
- **Values:** {1: male; 2: female}

### numeric__happy

- **numeric__happy** — Self-rated happiness on a 0–10 scale (scaled numeric feature).
- **Type:** Continuous
- **Example:** 0.332

### numeric__health

- **numeric__health** — Self-rated general health (1 very good to 5 very bad) (scaled numeric feature).
- **Type:** Ordinal
- **Example:** 0.941
- **Values:** {-0.162, -1.266, 0.941, 2.044, 3.148}

### numeric__height

- **numeric__height** — Self-reported height in centimeters (scaled numeric feature).
- **Type:** Continuous
- **Example:** 0.731

### numeric__hltprdi

- **numeric__hltprdi** — Doctor diagnosed diabetes (1 yes, 0 no) (scaled numeric feature).
- **Type:** Binary
- **Example:** -0.265
- **Values:** {1: yes; 0: no}

### numeric__hltprhb

- **numeric__hltprhb** — Doctor diagnosed high blood pressure (1 yes, 0 no) (scaled numeric feature).
- **Type:** Binary
- **Example:** 1.92
- **Values:** {1: yes; 0: no}

### numeric__inprdsc

- **numeric__inprdsc** — How often participates in organised social, religious, or community activities (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -1.208
- **Values:** {-0.505, -1.208, -1.911, 0.198, 0.902, 1.605, 2.308}

### numeric__paccnois

- **numeric__paccnois** — Perceived noise problems in the local area (1 yes, 0 no) (scaled numeric feature).
- **Type:** Binary
- **Example:** -0.221
- **Values:** {1: yes; 0: no}

### numeric__sclmeet

- **numeric__sclmeet** — Frequency of social meetings with friends, relatives, or colleagues (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.492
- **Values:** {-0.492, -1.123, -1.754, -2.384, 0.139, 0.769, 1.4}

### numeric__slprl

- **numeric__slprl** — How often sleep was restless in the last week (scaled numeric feature).
- **Type:** Ordinal
- **Example:** -0.937
- **Values:** {-0.937, 0.272, 1.481, 2.69}

### numeric__weighta

- **numeric__weighta** — Self-reported weight in kilograms (scaled numeric feature).
- **Type:** Continuous
- **Example:** 0.942

### numeric__wrhpp

- **numeric__wrhpp** — How often felt happy in the last week (reverse coded) (scaled numeric feature).
- **Type:** Ordinal
- **Example:** 0.141
- **Values:** {reverse: coded}
