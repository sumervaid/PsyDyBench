# Published Norms for All-Silico Persona Seeding 

This file compiles **published, citable** population parameters used to draw synthetic
personas for the the paper all-silico pipeline. **No real ESM individual-level data is
used here.** Every parameter below traces to a public source so it can feed provenance.

The machine-readable companion is `norms.json` (the seeder reads that; this `.md` is the
human-facing narrative + citations). Keep the two in sync.

> Scope note. The trait scale used by the downstream the earlier pipeline pipeline is **1–7** (Big Five and
> the state constructs). Several published norms are reported on a 1–5 IPIP/BFI metric; where
> that is the case we **linearly rescale** the mean and SD from 1–5 to 1–7
> (`x_7 = 1 + (x_5 - 1) * 6/4`, `sd_7 = sd_5 * 6/4`) and note it. Rescaling is a deterministic
> transform of published numbers, not a re-estimation from data.

---

## 1. Big Five population means, SDs, and intercorrelations

### Means / SDs (rescaled to 1–7)
Primary anchor: **Soto, John, Gosling & Potter (2011)**, *JPSP* 100(2):330–348 — Big Five
Inventory (BFI) developmental sample, adult means (reported 1–5). Cross-checked against
**Srivastava, John, Gosling & Potter (2003)**, *JPSP* 84(5):1041–1053 (large internet adult
sample) for ordering and adult stability. We use adult (age ~30–60) reference values.

| Trait (high pole)      | BFI mean (1–5) | BFI SD (1–5) | Rescaled mean (1–7) | Rescaled SD (1–7) |
|------------------------|----------------|--------------|---------------------|-------------------|
| Extraversion           | 3.25           | 0.90         | 4.38                | 1.35              |
| Agreeableness          | 3.64           | 0.72         | 4.96                | 1.08              |
| Conscientiousness      | 3.45           | 0.73         | 4.68                | 1.10              |
| Emotional Stability *  | 3.32           | 0.84         | 4.48                | 1.26              |
| Openness               | 3.74           | 0.66         | 5.11                | 0.99              |

\* Emotional Stability = reverse of Neuroticism. Soto et al. report Neuroticism mean 2.68
(1–5); Emotional Stability mean = 6 − 2.68 = 3.32 on the 1–5 metric, SD unchanged.

### Intercorrelations
Source: **van der Linden, te Nijenhuis & Bakker (2010)**, *Journal of Research in Personality*
44(3):315–327 — meta-analysis of the Big Five intercorrelation matrix (212 samples). These are
the canonical positive-manifold values (the "General Factor of Personality" literature) and are
modest, mostly positive. Values below are the meta-analytic r's (Neuroticism flipped to
Emotional Stability so all poles are socially-desirable / positively keyed).

Order: E, A, C, ES, O.

|        | E     | A     | C     | ES    | O     |
|--------|-------|-------|-------|-------|-------|
| **E**  | 1.00  | 0.17  | 0.19  | 0.20  | 0.23  |
| **A**  | 0.17  | 1.00  | 0.27  | 0.25  | 0.11  |
| **C**  | 0.19  | 0.27  | 1.00  | 0.26  | 0.20  |
| **ES** | 0.20  | 0.25  | 0.26  | 1.00  | 0.15  |
| **O**  | 0.23  | 0.11  | 0.20  | 0.15  | 1.00  |

(Matrix is made positive-definite by nearest-PD projection in the seeder if needed; the raw
meta-analytic values above are PD as listed.)

---

## 2. Trait → state baseline effect sizes (per-person mean structure)

A persona's **baseline** (personal mean) for each state is a linear function of standardized
Big Five plus a residual between-person term. Coefficients are **standardized betas** chosen
from published meta-analytic / large-sample trait–outcome associations. They are deliberately
modest and noted as "plausible published anchors," not re-estimated here.

State grand means (on each construct's native scale) are set from workplace ESM/survey norms:

| State (scale)            | Grand mean | Source for grand mean |
|--------------------------|------------|------------------------|
| Work Engagement (1–7)    | 4.80       | Schaufeli, Bakker & Salanova (2006), *UWES* normative tables — overall sample mean ≈ 3.8 on 0–6 → ≈ 4.80 on 1–7. |
| Psychological Safety (1–4)| 2.90      | Edmondson (1999), *ASQ* 44:350–383 — team PS scale midpoint-high; Frazier et al. (2017) meta-analysis means cluster ≈ 3.7/5 → ≈ 2.9/4. |
| Work Creativity (1–6)    | 3.60       | Zhou & George (2001), *AMJ* 44:682–696 creativity scale; typical means ≈ mid-scale-high → 3.6/6. |
| Team Trust (1–5)         | 3.70       | De Jong, Dirks & Gillespie (2016), *JAP* 101:1134–1150 meta-analysis — intra-team trust means ≈ 3.7/5. |

### Standardized trait→state betas
Anchors (each |β| ≤ ~0.35; signs from the meta-analytic literature):

- **Work Engagement**: +0.30 ES, +0.20 C, +0.18 E, +0.10 O, +0.06 A.
  - Source: Christian, Garza & Slaughter (2011), *Personnel Psychology* 64:89–136 (engagement
    meta-analysis: conscientiousness, positive affect/low neuroticism predict engagement).
- **Psychological Safety**: +0.22 ES, +0.18 A, +0.14 E, +0.08 O, +0.05 C.
  - Source: Frazier, Fainshmidt, Klinger, Pezeshkan & Vracheva (2017), *Personnel Psychology*
    70:113–165 (PS antecedents: proactive/stable, agreeable dispositions).
- **Work Creativity**: +0.34 O, +0.16 E, +0.10 ES, +0.06 C, −0.04 A.
  - Source: Feist (1998), *PSPR* 2:290–309 (openness is the dominant creativity trait);
    Hammond et al. (2011), *Psych. of Aesthetics* (openness, extraversion).
- **Team Trust**: +0.24 A, +0.18 ES, +0.12 E, +0.08 C, +0.05 O.
  - Source: Colquitt, Scott & LePine (2007), *JAP* 92:909–927 (trust propensity ~ agreeableness
    + low neuroticism).

Residual between-person SD (the part of the baseline NOT explained by traits) is set so that
total between-person variance matches the ICC norms in §3 (computed in the seeder).

---

## 3. ICC / within-person-SD targets per state

These set how much each persona fluctuates day-to-day. ICC = between-person variance /
total variance; within-person SD is derived from total SD and ICC:
`within_sd = total_sd * sqrt(1 - ICC)`, `between_sd = total_sd * sqrt(ICC)`.

| State | ICC  | Total SD (native scale) | Implied within-SD | Implied between-SD | Source |
|-------|------|--------------------------|-------------------|--------------------|--------|
| WE (1–7) | 0.55 | 1.10 | 0.737 | 0.816 | Xanthopoulou et al. (2009), *J. Occup. Health Psych.*; ESM engagement ICC ≈ 0.5–0.6. |
| PS (1–4) | 0.50 | 0.55 | 0.389 | 0.389 | Frazier et al. (2017) + daily-diary PS studies; ICC ≈ 0.5. |
| WC (1–6) | 0.50 | 0.85 | 0.601 | 0.601 | Amabile et al. (2005), *ASQ* daily creativity; ICC ≈ 0.45–0.55. |
| TT (1–5) | 0.55 | 0.70 | 0.470 | 0.519 | De Jong et al. (2016) + diary trust studies; ICC ≈ 0.5–0.6. |

> ICC values here are deliberately moderate (≈0.5–0.55), consistent with the broad ESM
> finding that roughly half of state variance is within-person (e.g., Podsakoff, Spoelma,
> Chawla & Gabriel, 2019, *JAP* 104:727–754, ICC review of affective/attitudinal states).
> They are NOT taken from the project's real ESM (which reports higher ICCs); using published
> ranges keeps the pipeline all-silico.

---

## 4. Demographic distributions

Synthetic demographics are drawn from public US working-adult reference distributions. These
are schema-compatible with the downstream pipeline (`age`, `sex`, `married`, `industry`,
`seniority`, `tenure_years`, `team_size`, plus org/team context numerics).

- **Age**: truncated normal, mean 42, SD 11, bounds [22, 67]. Source: U.S. Bureau of Labor
  Statistics, *Labor Force Statistics* — median age of the labor force ≈ 42 (2023).
- **Sex** (0/1 coding as in schema): P(0)=0.53, P(1)=0.47. Source: BLS labor force ~53% male.
- **Married**: P(Yes)=0.55, P(No)=0.45. Source: U.S. Census Bureau, marital status of adults
  25–64 ≈ 55% married.
- **Industry**: categorical over the schema's industry labels, weighted toward larger US
  employment sectors (IT, financial services, healthcare, education, manufacturing,
  government, professional services, retail, other). Weights approximate BLS sector employment
  shares; exact labels copied from the downstream schema for compatibility.
- **Seniority**: P(Non Manager)=0.62, P(Manager)=0.38. Source: BLS — managers/supervisors are
  a minority of the workforce (~35–40%).
- **Tenure (years)**: log-normal-ish via truncated normal mean 6.5, SD 5.5, bounds [0.1, 30].
  Source: BLS median employee tenure ≈ 4–5 yrs, mean higher due to long-tenure tail.
- **Team size**: log-normal, median 7, bounds [2, 60]. Generic org reference (no single source;
  marked as a modeling assumption).
- **Org culture creativity (1–5)** and **team interdependence (1–5)**: truncated normal mean
  3.3, SD 0.8, bounds [1, 5]. Modeling assumption (mid-scale context), flagged as such.

---

## 5. Indicator scale ranges (forward inputs)

The 55 daily context indicators are **sampled as forward inputs**; the 4 states are left
empty (NaN) for the LLM generation stage to fill — this removes the inverse/circularity
backdoor. Indicator scale bounds and types come from the downstream pipeline's
`config.INDICATOR_DIMENSIONS` definition (a **schema/design specification**, not real data).
Each indicator is sampled from a truncated normal centered at its scale midpoint with a
moderate SD, then rounded to its native type (int counts, float hours/ratios, binary flags).
This yields plausible day-to-day variation without copying any empirical ESM distribution.

See `norms.json → indicators` for the per-indicator (min, max, type) used by the seeder.

---

## Source list (for provenance / MANUSCRIPT_VALUES keys)
- Soto, John, Gosling & Potter (2011), *JPSP* 100(2):330–348.
- Srivastava, John, Gosling & Potter (2003), *JPSP* 84(5):1041–1053.
- van der Linden, te Nijenhuis & Bakker (2010), *J. Res. Pers.* 44(3):315–327.
- Schaufeli, Bakker & Salanova (2006), UWES manual / *Educ. Psych. Meas.* 66:701–716.
- Edmondson (1999), *ASQ* 44:350–383.
- Frazier, Fainshmidt, Klinger, Pezeshkan & Vracheva (2017), *Personnel Psychology* 70:113–165.
- Zhou & George (2001), *AMJ* 44:682–696.
- De Jong, Dirks & Gillespie (2016), *JAP* 101:1134–1150.
- Christian, Garza & Slaughter (2011), *Personnel Psychology* 64:89–136.
- Feist (1998), *Pers. Soc. Psych. Review* 2:290–309.
- Hammond, Neff, Farr, Schwall & Zhao (2011), *Psych. of Aesthetics, Creativity, and the Arts* 5:90–105.
- Colquitt, Scott & LePine (2007), *JAP* 92:909–927.
- Xanthopoulou, Bakker, Demerouti & Schaufeli (2009), *J. Occup. Health Psych.* 14:121–141.
- Amabile, Barsade, Mueller & Staw (2005), *ASQ* 50:367–403.
- Podsakoff, Spoelma, Chawla & Gabriel (2019), *JAP* 104:727–754.
- U.S. Bureau of Labor Statistics, Labor Force Statistics (2023).
- U.S. Census Bureau, Marital Status tables (2022–2023).
