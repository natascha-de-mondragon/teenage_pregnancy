# Statistical Modelling of Teenage Pregnancy Determinants in Uganda  

---

## Overview  
This repository contains the statistical modelling and data analysis supporting the paper:  

**“How much of a change in proximal determinants of adolescent pregnancy is needed to reduce teenage pregnancies in Uganda?  
A decomposition analysis of the 2018 and 2023 Adolescent Girls and Young Women Survey Data.”**

The analysis examines the extent to which changes in *proximal determinants*—such as **schooling status, age at sexual debut, marriage timing, and contraceptive use**—contribute to reductions in teenage pregnancy in Uganda.

---

## Background  
Teenage pregnancy remains a significant social and public health issue in Uganda. According to the 2022 Uganda Demographic and Health Survey (UDHS), **23.5% of women aged 15–19 had initiated childbearing**, with marked disparities by education level, residence, and region.  

While many studies describe associated factors, few quantify how much improvement in these determinants could reduce adolescent pregnancy. This analysis uses decomposition and regression methods to assess the contribution of key factors driving teenage pregnancy among adolescent girls and young women (AGYW) in Uganda.

---

## Data Source  
The analysis uses data from the **2018 and 2023 Adolescent Girls and Young Women (AGYW) Survey**, conducted across 20 districts by **Makerere University School of Public Health (MakSPH)** and partners.  
The corresponding **survey questionnaire** is provided in this repository (`survey_outline.pdf`) for reference.

### Sampling Design  
- **Population:** Adolescent girls and young women aged 10–24 years (N = 8,473)  
- **Sampling Frame:** 233 villages and 80 schools in 20 purposively selected districts  
- **Stratification:** In-school and out-of-school participants  
- **Variables:** Socio-demographic characteristics, reproductive health, sexual behaviour, contraceptive use, pregnancy history, and household vulnerability  

---

## Analytical Approach  
The notebook applies both **descriptive and inferential statistical analyses** in Python.  

**Analytical methods include:**  
- Descriptive statistics: frequencies, proportions, cross-tabulations  
- Chi-square tests for group comparisons  
- Binary logistic regression to assess associations between teenage pregnancy and explanatory factors  
- Kaplan-Meier survival analysis to estimate time to first pregnancy following sexual debut  
- Decomposition analysis to determine the contribution of proximal determinants (education, marriage, contraception, sexual debut) to overall reductions in teenage pregnancy  

---

## Key Variables  
| Category | Example Variables |
|-----------|------------------|
| Socio-demographic | Age, residence (rural/urban), district |
| Education | Schooling status (in-school, completed, dropped out), highest level completed |
| Sexual behaviour | Age at sexual debut, willingness, partner type |
| Contraception | Use at first sex, type of method |
| Marriage | Marital status at pregnancy |
| Wealth | Household asset-based tertiles (PCA derived) |

---

## Outputs  
Below are selected visualizations from the analysis (available in the `results/` directory):

**1. Mean Time to Pregnancy by Age at Sexual Debut**  
Shows how the average time from sexual debut to first pregnancy shortens with earlier sexual initiation.  
![Mean Time to Pregnancy by Age at Sexual Debut](results/mean_time_to_preg_by_debut_age_errorbars.png)

**2. Kaplan–Meier Curve: Time to Pregnancy After Sexual Debut**  
Estimates the probability of remaining non-pregnant over time after sexual initiation.  
![Kaplan–Meier Time to Pregnancy](results/km_time_to_preg.png)

**3. Kaplan–Meier Curve: Time to Pregnancy After Marriage**  
Shows cumulative probability of pregnancy following marriage among adolescent girls.  
![Kaplan–Meier Time to Pregnancy After Marriage](results/km_time_to_preg_after_marry.png)

**4. Global Dumbbell Plot: Sexual Debut vs. Pregnancy Timing**  
Compares mean ages at sexual debut and at first pregnancy, illustrating exposure windows and interquartile ranges.  
![Dumbbell Plot — Debut vs. Pregnancy Timing](results/dumbbell_debut_vs_preg_GLOBAL_IQR.png)

---

## Tools and Packages  
Developed using **Python 3.10+** with the following libraries:  
- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `statsmodels`  
- `lifelines`  

---

## Funding Acknowledgement  
This study was supported by a **grant from the Global Fund through The AIDS Support Organization (TASO)**  
(**Grant#: UGA-C-TASO-1449**) awarded to **Makerere University School of Public Health** to conduct formative research on HIV, sexual and reproductive health, and gender-based violence among adolescent girls and young women in Uganda.

---

## Citation  
If using or referencing this work, please cite:  

> Matovu, J.K.B., Minnitt, N.J., et al. (in preparation).  
> *How much of a change in proximal determinants of adolescent pregnancy is needed to reduce teenage pregnancies in Uganda?*  
> Makerere University School of Public Health.

---

## License  
This repository is made available for academic and non-commercial use.  
Please credit the authors and **Makerere University School of Public Health** when reproducing content or analytical methods.

---
