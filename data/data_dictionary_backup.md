# Data Dictionary — Health_XAI_Project

## Overview
This document describes the variables available in `data/raw/heart_data.csv` after preprocessing and feature-name standardisation. Use it as a reference when building models, generating explanations (LIME/SHAP), and drafting project reports.

## Index of Features

[categorical__cntry_AT](#categorical__cntry_at) · [categorical__cntry_BE](#categorical__cntry_be) · [categorical__cntry_BG](#categorical__cntry_bg) · [categorical__cntry_CH](#categorical__cntry_ch) · [categorical__cntry_CY](#categorical__cntry_cy) · [categorical__cntry_DE](#categorical__cntry_de) · [categorical__cntry_ES](#categorical__cntry_es) · [categorical__cntry_FI](#categorical__cntry_fi) · [categorical__cntry_FR](#categorical__cntry_fr) · [categorical__cntry_GB](#categorical__cntry_gb) · [categorical__cntry_GR](#categorical__cntry_gr) · [categorical__cntry_HR](#categorical__cntry_hr) · [categorical__cntry_HU](#categorical__cntry_hu) · [categorical__cntry_IE](#categorical__cntry_ie) · [categorical__cntry_IL](#categorical__cntry_il) · [categorical__cntry_IS](#categorical__cntry_is) · [categorical__cntry_IT](#categorical__cntry_it) · [categorical__cntry_LT](#categorical__cntry_lt) · [categorical__cntry_LV](#categorical__cntry_lv) · [categorical__cntry_ME](#categorical__cntry_me) · [categorical__cntry_NL](#categorical__cntry_nl) · [categorical__cntry_NO](#categorical__cntry_no) · [categorical__cntry_PL](#categorical__cntry_pl) · [categorical__cntry_PT](#categorical__cntry_pt) · [categorical__cntry_RS](#categorical__cntry_rs) · [categorical__cntry_SE](#categorical__cntry_se) · [categorical__cntry_SI](#categorical__cntry_si) · [categorical__cntry_SK](#categorical__cntry_sk) · [hltprhc](#hltprhc) · [numeric__alcfreq](#numeric__alcfreq) · [numeric__cgtsmok](#numeric__cgtsmok) · [numeric__ctrlife](#numeric__ctrlife) · [numeric__dosprt](#numeric__dosprt) · [numeric__eatveg](#numeric__eatveg) · [numeric__enjlf](#numeric__enjlf) · [numeric__etfruit](#numeric__etfruit) · [numeric__fltdpr](#numeric__fltdpr) · [numeric__flteeff](#numeric__flteeff) · [numeric__fltlnl](#numeric__fltlnl) · [numeric__fltsd](#numeric__fltsd) · [numeric__gndr](#numeric__gndr) · [numeric__happy](#numeric__happy) · [numeric__health](#numeric__health) · [numeric__height](#numeric__height) · [numeric__hltprdi](#numeric__hltprdi) · [numeric__hltprhb](#numeric__hltprhb) · [numeric__inprdsc](#numeric__inprdsc) · [numeric__paccnois](#numeric__paccnois) · [numeric__sclmeet](#numeric__sclmeet) · [numeric__slprl](#numeric__slprl) · [numeric__weighta](#numeric__weighta) · [numeric__wrhpp](#numeric__wrhpp)

## Feature Table

| Feature | Description | Type | Example |
|---------|-------------|------|---------|
| categorical__cntry_AT | cntry == AT (one-hot encoded). | Binary | 1 |
| categorical__cntry_BE | cntry == BE (one-hot encoded). | Binary | 0 |
| categorical__cntry_BG | cntry == BG (one-hot encoded). | Binary | 0 |
| categorical__cntry_CH | cntry == CH (one-hot encoded). | Binary | 0 |
| categorical__cntry_CY | cntry == CY (one-hot encoded). | Binary | 0 |
| categorical__cntry_DE | cntry == DE (one-hot encoded). | Binary | 0 |
| categorical__cntry_ES | cntry == ES (one-hot encoded). | Binary | 0 |
| categorical__cntry_FI | cntry == FI (one-hot encoded). | Binary | 0 |
| categorical__cntry_FR | cntry == FR (one-hot encoded). | Binary | 0 |
| categorical__cntry_GB | cntry == GB (one-hot encoded). | Binary | 0 |
| categorical__cntry_GR | cntry == GR (one-hot encoded). | Binary | 0 |
| categorical__cntry_HR | cntry == HR (one-hot encoded). | Binary | 0 |
| categorical__cntry_HU | cntry == HU (one-hot encoded). | Binary | 0 |
| categorical__cntry_IE | cntry == IE (one-hot encoded). | Binary | 0 |
| categorical__cntry_IL | cntry == IL (one-hot encoded). | Binary | 0 |
| categorical__cntry_IS | cntry == IS (one-hot encoded). | Binary | 0 |
| categorical__cntry_IT | cntry == IT (one-hot encoded). | Binary | 0 |
| categorical__cntry_LT | cntry == LT (one-hot encoded). | Binary | 0 |
| categorical__cntry_LV | cntry == LV (one-hot encoded). | Binary | 0 |
| categorical__cntry_ME | cntry == ME (one-hot encoded). | Binary | 0 |
| categorical__cntry_NL | cntry == NL (one-hot encoded). | Binary | 0 |
| categorical__cntry_NO | cntry == NO (one-hot encoded). | Binary | 0 |
| categorical__cntry_PL | cntry == PL (one-hot encoded). | Binary | 0 |
| categorical__cntry_PT | cntry == PT (one-hot encoded). | Binary | 0 |
| categorical__cntry_RS | cntry == RS (one-hot encoded). | Binary | 0 |
| categorical__cntry_SE | cntry == SE (one-hot encoded). | Binary | 0 |
| categorical__cntry_SI | cntry == SI (one-hot encoded). | Binary | 0 |
| categorical__cntry_SK | cntry == SK (one-hot encoded). | Binary | 0 |
| hltprhc | *[Description pending clarification]* | Binary | 0 |
| numeric__alcfreq | *[Description pending clarification]* | Ordinal | -0.795 |
| numeric__cgtsmok | *[Description pending clarification]* | Ordinal | -0.19 |
| numeric__ctrlife | *[Description pending clarification]* | Continuous | 0.31 |
| numeric__dosprt | *[Description pending clarification]* | Ordinal | -0.061 |
| numeric__eatveg | *[Description pending clarification]* | Ordinal | -0.142 |
| numeric__enjlf | *[Description pending clarification]* | Ordinal | 0.164 |
| numeric__etfruit | *[Description pending clarification]* | Ordinal | -0.172 |
| numeric__fltdpr | *[Description pending clarification]* | Ordinal | -0.661 |
| numeric__flteeff | *[Description pending clarification]* | Ordinal | -0.863 |
| numeric__fltlnl | *[Description pending clarification]* | Ordinal | -0.61 |
| numeric__fltsd | *[Description pending clarification]* | Ordinal | -0.842 |
| numeric__gndr | *[Description pending clarification]* | Binary | -1.053 |
| numeric__happy | *[Description pending clarification]* | Continuous | 0.332 |
| numeric__health | *[Description pending clarification]* | Ordinal | 0.941 |
| numeric__height | *[Description pending clarification]* | Continuous | 0.731 |
| numeric__hltprdi | *[Description pending clarification]* | Binary | -0.265 |
| numeric__hltprhb | *[Description pending clarification]* | Binary | 1.92 |
| numeric__inprdsc | *[Description pending clarification]* | Ordinal | -1.208 |
| numeric__paccnois | *[Description pending clarification]* | Binary | -0.221 |
| numeric__sclmeet | *[Description pending clarification]* | Ordinal | -0.492 |
| numeric__slprl | *[Description pending clarification]* | Ordinal | -0.937 |
| numeric__weighta | *[Description pending clarification]* | Continuous | 0.942 |
| numeric__wrhpp | *[Description pending clarification]* | Ordinal | 0.141 |

## Detailed Feature Notes

### categorical__cntry_AT
- **categorical__cntry_AT** — cntry == AT (one-hot encoded).
- **Type:** Binary
- **Example:** 1
- **Values:** {0: Not AT; 1: AT}

### categorical__cntry_BE
- **categorical__cntry_BE** — cntry == BE (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not BE; 1: BE}

### categorical__cntry_BG
- **categorical__cntry_BG** — cntry == BG (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not BG; 1: BG}

### categorical__cntry_CH
- **categorical__cntry_CH** — cntry == CH (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not CH; 1: CH}

### categorical__cntry_CY
- **categorical__cntry_CY** — cntry == CY (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not CY; 1: CY}

### categorical__cntry_DE
- **categorical__cntry_DE** — cntry == DE (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not DE; 1: DE}

### categorical__cntry_ES
- **categorical__cntry_ES** — cntry == ES (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not ES; 1: ES}

### categorical__cntry_FI
- **categorical__cntry_FI** — cntry == FI (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not FI; 1: FI}

### categorical__cntry_FR
- **categorical__cntry_FR** — cntry == FR (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not FR; 1: FR}

### categorical__cntry_GB
- **categorical__cntry_GB** — cntry == GB (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not GB; 1: GB}

### categorical__cntry_GR
- **categorical__cntry_GR** — cntry == GR (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not GR; 1: GR}

### categorical__cntry_HR
- **categorical__cntry_HR** — cntry == HR (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not HR; 1: HR}

### categorical__cntry_HU
- **categorical__cntry_HU** — cntry == HU (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not HU; 1: HU}

### categorical__cntry_IE
- **categorical__cntry_IE** — cntry == IE (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IE; 1: IE}

### categorical__cntry_IL
- **categorical__cntry_IL** — cntry == IL (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IL; 1: IL}

### categorical__cntry_IS
- **categorical__cntry_IS** — cntry == IS (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IS; 1: IS}

### categorical__cntry_IT
- **categorical__cntry_IT** — cntry == IT (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not IT; 1: IT}

### categorical__cntry_LT
- **categorical__cntry_LT** — cntry == LT (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not LT; 1: LT}

### categorical__cntry_LV
- **categorical__cntry_LV** — cntry == LV (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not LV; 1: LV}

### categorical__cntry_ME
- **categorical__cntry_ME** — cntry == ME (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not ME; 1: ME}

### categorical__cntry_NL
- **categorical__cntry_NL** — cntry == NL (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not NL; 1: NL}

### categorical__cntry_NO
- **categorical__cntry_NO** — cntry == NO (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not NO; 1: NO}

### categorical__cntry_PL
- **categorical__cntry_PL** — cntry == PL (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not PL; 1: PL}

### categorical__cntry_PT
- **categorical__cntry_PT** — cntry == PT (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not PT; 1: PT}

### categorical__cntry_RS
- **categorical__cntry_RS** — cntry == RS (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not RS; 1: RS}

### categorical__cntry_SE
- **categorical__cntry_SE** — cntry == SE (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not SE; 1: SE}

### categorical__cntry_SI
- **categorical__cntry_SI** — cntry == SI (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not SI; 1: SI}

### categorical__cntry_SK
- **categorical__cntry_SK** — cntry == SK (one-hot encoded).
- **Type:** Binary
- **Example:** 0
- **Values:** {0: Not SK; 1: SK}

### hltprhc
- **hltprhc** — *[Description pending clarification]*
- **Type:** Binary
- **Example:** 0
- **Values:** {0, 1}

### numeric__alcfreq
- **numeric__alcfreq** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.795
- **Values:** {-0.306, -0.795, -1.284, -1.773, 0.183, 0.673, 1.162}

### numeric__cgtsmok
- **numeric__cgtsmok** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.19
- **Values:** {-0.19, -0.737, -1.284, -1.831, 0.357, 0.904}

### numeric__ctrlife
- **numeric__ctrlife** — *[Description pending clarification]*
- **Type:** Continuous
- **Example:** 0.31

### numeric__dosprt
- **numeric__dosprt** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.061
- **Values:** {-0.061, -0.447, -0.834, -1.22, 0.325, 0.712, 1.098, 1.485}

### numeric__eatveg
- **numeric__eatveg** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.142
- **Values:** {-0.142, -1.041, -1.941, 0.758, 1.657, 2.557, 3.457}

### numeric__enjlf
- **numeric__enjlf** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** 0.164
- **Values:** {-1.01, -2.184, 0.164, 1.338}

### numeric__etfruit
- **numeric__etfruit** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.172
- **Values:** {-0.172, -0.937, -1.702, 0.593, 1.358, 2.123, 2.888}

### numeric__fltdpr
- **numeric__fltdpr** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.661
- **Values:** {-0.661, 0.843, 2.346, 3.85}

### numeric__flteeff
- **numeric__flteeff** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.863
- **Values:** {-0.863, 0.404, 1.67, 2.937}

### numeric__fltlnl
- **numeric__fltlnl** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.61
- **Values:** {-0.61, 0.788, 2.186, 3.584}

### numeric__fltsd
- **numeric__fltsd** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.842
- **Values:** {-0.842, 0.634, 2.11, 3.586}

### numeric__gndr
- **numeric__gndr** — *[Description pending clarification]*
- **Type:** Binary
- **Example:** -1.053
- **Values:** {-1.053, 0.95}

### numeric__happy
- **numeric__happy** — *[Description pending clarification]*
- **Type:** Continuous
- **Example:** 0.332

### numeric__health
- **numeric__health** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** 0.941
- **Values:** {-0.162, -1.266, 0.941, 2.044, 3.148}

### numeric__height
- **numeric__height** — *[Description pending clarification]*
- **Type:** Continuous
- **Example:** 0.731

### numeric__hltprdi
- **numeric__hltprdi** — *[Description pending clarification]*
- **Type:** Binary
- **Example:** -0.265
- **Values:** {-0.265, 3.777}

### numeric__hltprhb
- **numeric__hltprhb** — *[Description pending clarification]*
- **Type:** Binary
- **Example:** 1.92
- **Values:** {-0.521, 1.92}

### numeric__inprdsc
- **numeric__inprdsc** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -1.208
- **Values:** {-0.505, -1.208, -1.911, 0.198, 0.902, 1.605, 2.308}

### numeric__paccnois
- **numeric__paccnois** — *[Description pending clarification]*
- **Type:** Binary
- **Example:** -0.221
- **Values:** {-0.221, 4.532}

### numeric__sclmeet
- **numeric__sclmeet** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.492
- **Values:** {-0.492, -1.123, -1.754, -2.384, 0.139, 0.769, 1.4}

### numeric__slprl
- **numeric__slprl** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** -0.937
- **Values:** {-0.937, 0.272, 1.481, 2.69}

### numeric__weighta
- **numeric__weighta** — *[Description pending clarification]*
- **Type:** Continuous
- **Example:** 0.942

### numeric__wrhpp
- **numeric__wrhpp** — *[Description pending clarification]*
- **Type:** Ordinal
- **Example:** 0.141
- **Values:** {-1.1, -2.34, 0.141, 1.381}
