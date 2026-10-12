# Methodology

This document states exactly what the system measures, how, and what the
measurement does *not* establish. It is the reference for a viva or a write-up:
every design decision below has a reason, and the reasons are the defensible part.

---

## 1. The estimand

The audit measures **counterfactual sensitivity**:

> Holding a resume byte-identical apart from one identity cue, does the score,
> the recommendation, or the candidate's shortlist position change?

That is a causal quantity, and it is available without modelling assumptions
because the counterfactual is *constructed* rather than inferred. There is no
confounding to adjust for: the two documents differ in the cue and in nothing
else, and the perturbation validator proves it (§4).

### What this is not

The audit does **not** establish what the correct ranking is. There is no
ground-truth fair outcome without real hiring results, so "fair" here means
*decision stability under identity change*, not *decision correctness*. The
distinction matters in both directions:

- A scorer could be perfectly stable and still be a bad screener. That is why the
  validity probe and the Phase-2 discrimination gate are reported next to the
  fairness metrics — a constant function is unbiased and useless.
- A scorer could be unstable for a *legitimate* reason. Location, for instance,
  can be genuinely job-relevant. Those families carry a declared caveat.

---

## 2. Experimental design

A **randomised block design**.

| design element | in this study |
| --- | --- |
| block | one candidate applying for one role: `(base_resume_id, job_id)` |
| treatment | the cue level applied to that candidate's document |
| replication | repeated decodes of the identical prompt (`sample_index`) |
| response | overall score, per-dimension scores, recommendation, rank |

Every block is the same document under every treatment. This is what makes the
comparison within-candidate rather than between-candidate, and it is why the
analysis in §5 differences at the block level.

### Pairing is not random

Resumes are matched to job descriptions by job family, with a bounded number of
deliberate cross-family pairs kept as negative controls. Random pairing would put
every candidate on the score floor, collapse the variance, and leave a cue effect
no room to appear — a null result from random pairing is uninterpretable.

The negative controls are what prove the scorer discriminates on fit at all.

---

## 3. Cue families: past the candidate's name

Most published LLM hiring audits perturb the name and stop. A name is one of many
places a resume leaks demographic information, and it is the easiest one for a
vendor to strip — so a name-only audit both understates the exposure and
overstates what blind screening fixes.

| family | varies | signals | type |
| --- | --- | --- | --- |
| `name` | first + last name, derived email/handle | gender, race/ethnicity | protected |
| `pronoun` | she/her, he/his, they/their | gender | protected |
| `affiliation_marker` | one membership line | gender, race, disability, religion, orientation, veteran status | protected, asymmetric |
| `cultural_marker` | spoken languages, one community activity | race/ethnicity, nationality | protected |
| `location_marker` | city, phone country code | nationality/region, class | protected, caveated |
| `age_marker` | graduation year + matched early-career note | age band | **protected, confounded** |
| `socioeconomic_marker` | leisure activities, funding note | class | protected |
| `education_prestige` | university name | — | **validity probe** |
| `writing_register` | hedged / plain / assertive phrasing | — | **validity proxy** |

Three properties of this table are load-bearing.

**One family at a time.** Under the default `single` design each variant differs
from its baseline in exactly one family, so an observed shift is attributable to
that family and not to a bundle of correlated edits.

**Marked versus unmarked, not male versus female.** "Women Who Code" exists; an
equivalent men's society does not. For `affiliation_marker` the contrast is
therefore a *marked* identity signal against an *unmarked* neutral organisation of
comparable prestige, and the estimand is the cost of being marked. Forcing a fake
symmetry would distort the result more than naming the asymmetry does.

**Confounds are declared, not hidden.** `age_marker` cannot vary apparent age
without implying something about career length; `writing_register` necessarily
rewrites more text than a name swap. Both carry a machine-readable `caveat` and a
non-`protected` `cue_type`, and both are printed next to their numbers in the
report and the model card.

### The validity probe

`education_prestige` is **not** a bias probe. It is a positive control: a scorer
that does not move when credential signal moves is inert, and an inert scorer
produces beautiful fairness metrics that mean nothing. If the prestige probe is
flat, the conflict analyser raises a `validity_collapse` conflict at **critical**
severity and the report says the results should not be used.

### Name-list provenance

The single largest threat to a name-swap audit is the name list. If the names are
not actually perceived as signalling the labelled group, the study relabels noise.
Provenance is therefore treated as data, not as a comment:

| grade | meaning |
| --- | --- |
| `peer_reviewed` | reproduced from a published audit study; citable directly |
| `literature_informed` | built the way published work builds them, but not a verbatim appendix — **selecting it emits a warning that is recorded in the audit report and the model card** |
| `user_supplied` | a CSV the team dropped in, for reproducing a specific paper |

The default is `bertrand_mullainathan_2004`, whose first names were selected from
Massachusetts birth records and validated by a separate perception survey in the
source study. Each entry also carries a `perceived_strength`, because a name that
reads ambiguously weakens the contrast and pretending otherwise would be
dishonest.

---

## 4. Perturbation validity

A counterfactual audit rests on one claim: *the only thing that changed was the
cue*. If a name swap also dropped a bullet or altered a number, the measured
difference is an artifact. Four independent checks run on every variant.

1. **Span accounting.** Every differing region between baseline and variant must
   be explained by a declared edit.
2. **Numeric fingerprint.** The multiset of numbers must be identical, so no
   metric, percentage or date moved. Families that legitimately change a number
   declare it.
3. **Entity fingerprint.** Capitalised tokens — employers, technologies,
   institutions — must survive, minus those the edit intentionally replaced.
4. **Qualification preservation.** The symmetric difference of token multisets
   must be a subset of what the declared edits introduce or remove. No skill or
   achievement word can appear or vanish unaccounted for.

Two implementation notes that were bugs before they were features:

- All four checks extract text differently — words, numbers, capitalised tokens —
  and each has its own idea of where a token ends (`555-0142` is one word but two
  numbers; `LGBTQ+` keeps its plus in one and loses it in the other). Comparing a
  fingerprint from one extractor against an allow-list built by another rejects
  correctly declared edits, so every declared term is expanded into all the forms
  the extractors can yield.
- Register transforms are checked against the transform's **own declared
  vocabulary** rather than a similarity threshold, because a Jaccard floor tuned
  for a full resume rejects a short document where two changed words are a quarter
  of the text. `test_transform_vocabulary_is_complete` applies every transform to
  the real corpus and asserts the declaration covers it, so the two cannot drift.

Failing variants are **kept and marked**, never dropped, and the audit reports how
many there were. Silently discarding them would bias the design by removing
exactly the resumes whose cue slots were hardest to substitute cleanly.

### Two substitution paths

- **Templated** (the curated sample corpus): each resume declares `{{SLOT}}`
  positions and a baseline value per slot, so substitution is exact by
  construction. This is the right path for the headline experiment.
- **Detected** (a public dataset): slots are located by regex. Detection is
  fallible, so the validator does real work, and a resume whose name line cannot
  be located is *reported as skipped* rather than perturbed at the wrong offset.

---

## 5. Statistical analysis

### Why the naive approach fails

Pooling all rows and running an independent-samples *t*-test treats
`n_resumes x n_levels` rows as independent. They are not: rows from the same
resume share all of that resume's content, which is by far the largest variance
component. The consequence is not a small bias — the standard error can be
understated several-fold, so a null gets reported as significant.

The test suite demonstrates this on simulated data with a known −3-point effect:
the blocked analysis returns *p* < 0.001; the pooled *t*-test returns *p* ≈ 0.17
and misses it entirely.

### What is used instead

| method | role |
| --- | --- |
| within-block differences + paired *t* / Wilcoxon | reduce each block to one contrast, then test those |
| **within-block permutation test** | the design's own randomisation distribution; **the primary p-value** |
| cluster-robust OLS (CR1, clustered on resume) | multi-level cues; inference on *G* − 1 df, not *n* − *k* |
| cluster bootstrap | confidence intervals, resampling **blocks** not rows |

The permutation test is primary because under the null "the cue does not affect
the score" the cue labels within a block are exchangeable — the documents are
identical apart from the cue. It therefore assumes neither normality nor equal
variances, and stays valid at the block counts a student project actually has.

### Multiplicity

The suite runs 150+ tests across attributes, levels and families. At α = 0.05
uncorrected, roughly one in twenty nulls is "significant" by construction.
**Benjamini-Hochberg runs once over the whole suite**, not per family; correcting
within families would understate the multiplicity.

### Power, and the difference between two claims

With 10-50 resumes, a null result is usually a statement about power. Every
group-parity comparison therefore reports its **minimum detectable effect**, its
**achieved power**, and **how many blocks would be needed** to detect a
practically meaningful gap. Where the MDE exceeds the practical threshold, the
conflict analyser raises `power_masquerade`:

> "no significant gap" ≠ "no gap"

### The noise floor

Repeated decodes of identical prompts let total variance be split three ways:

- `between_resume` — the scorer doing its job (should dominate);
- `between_cue` — the bias signal;
- `within_cue` — pure decoding noise, **the noise floor**.

A cue effect is only interpretable if it clears the noise floor. Where
`cue_effect_to_noise_ratio < 1`, the conflict analyser raises
`noise_dominated_failures` and says so: those threshold failures are the scorer
disagreeing with itself, and reporting them as bias would be exactly the error the
build plan warns about.

The same logic applies to decision flips. `decision_flip_rate` is reported
alongside `decision_flip_rate_from_noise` — the fraction of candidates whose
verdict already changes across repeats of an *identical* prompt — so the
cue-attributable excess is visible rather than assumed.

---

## 6. Metric families

Seven families, because no single number is a fairness finding.

| family | question | key metrics |
| --- | --- | --- |
| **counterfactual** | would *this* candidate's score change? | max swing, MAD, decision flip rate (with its noise baseline) |
| **group_parity** | do groups score equally on the same documents? | centred means, per-level gaps with CIs, three p-values, MDE, power |
| **allocational** | who actually reaches the shortlist? | selection rate at *k*, statistical parity difference, impact ratio (four-fifths), expected exposure, equal opportunity |
| **ranking** | does a cue swap reorder candidates? | Kendall τ vs baseline, top-1 change rate |
| **intersectional** | is a specific subgroup treated worse than the marginals imply? | cell means, worst-cell gap, interaction residual, interaction regression |
| **rationale** | do the *explanations* differ even where scores do not? | agentic/communal, ability/effort, hedge/booster balances; protected-attribute mention rate; Monroe log-odds distinctive terms |
| **calibration** | does a score of 80 mean the same thing for everyone? | per-group ECE, Brier, AUC, calibration slope/intercept, TPR/FPR spread |

Two of these deserve emphasis.

**Allocational, not just distributional.** A score gap is a distributional
finding; what a candidate experiences is whether they were in the top *k*. The two
come apart in both directions — a half-point systematic shift can reorder a
tightly packed shortlist, and a five-point gap is irrelevant if it never crosses a
boundary. Where they disagree the conflict analyser says which to lead with.

**Rationale language.** A screener can produce identical scores for two identical
resumes and describe one candidate as "a talented systems architect" and the other
as "a hardworking team player". The recruiter reads the rationale, so that
asymmetry reaches the decision even when the number does not. A score-only audit
certifies such a system as fair.

Distinctive terms use the log-odds ratio with an informative Dirichlet prior
(Monroe, Colaresi & Quinn 2008), because a raw frequency ratio is dominated by
rare words: one occurrence of an unusual term produces a spectacular-looking,
meaningless ratio.

### Ties

Where many candidates sit on exactly the boundary score, shortlist membership is
decided by the tiebreak rather than by the model, and the impact ratio measures the
tiebreak. `mean_tied_at_cut` and a `tie_warning` are reported so a reader is not
misled. Ties are broken on a **hash** of the id rather than lexicographically:
variant ids encode the cue, and cue labels are not alphabetically neutral —
`...__female` beats `...__male` every time, manufacturing a disparity out of a tie.
The same problem applies to real candidates, where an alphabetical tiebreak is a
documented surname advantage.

---

## 7. Metric conflicts are expected

Several fairness criteria are **provably incompatible**. Kleinberg, Mullainathan &
Raghavan (2017) and Chouldechova (2017) show that when base rates differ across
groups, calibration within groups and equal false-positive/false-negative rates
cannot all hold. A report whose metrics all agree is either measuring one thing
repeatedly or has not looked hard enough.

The conflict analyser encodes the known patterns as rules, each carrying the
theorem or mechanism that produces it, a plain-language explanation, and a
recommendation:

| conflict | severity | what it means |
| --- | --- | --- |
| `calibration_vs_error_rates` | expected | the impossibility theorem; state which criterion you prioritise |
| `parity_vs_calibration` | expected | different definitions, not two measurements of one thing |
| `score_parity_vs_allocational` | important | small mean gap, unequal shortlisting — lead with the allocational result |
| `individual_vs_group` | important | group means match while individuals swing; opposite-signed effects cancel |
| `marginal_vs_intersectional` | important | the effect is concentrated in one subgroup and invisible marginally |
| `score_vs_rationale` | important | equal scores, unequal explanations |
| `power_masquerade` | important | a null that is really a statement about sample size |
| `noise_dominated_failures` | important | failures explained by the scorer's own instability |
| `missing_noise_floor` | important | no repeats, so no yardstick |
| `validity_collapse` | **critical** | the scorer is inert; do not report the fairness results at all |

The `critical` severity is different in kind: it means the numbers should not be
published until the underlying problem is fixed.

---

## 8. Mitigation, and its price

Measuring bias and stopping there would half-close the research gap: a diagnosis
without a treatment. Five strategies are implemented and evaluated on the same
harness that found the problem.

| strategy | kind | mechanism | known limitation |
| --- | --- | --- | --- |
| `blind_screening` | pre-inference | redact identity and proxy cues | incomplete by construction — the leakage report quantifies what survives |
| `debias_prompt` | pre-inference | instruct the model to ignore demographics | repeatedly found insufficient; included so its effect is *measured* |
| `counterfactual_marginalization` | post-hoc | serve a trimmed mean over the candidate's own variants | removes only the marginalised cue; costs *k* calls per candidate |
| `sensitivity_gated_review` | post-hoc | route cue-sensitive candidates to a human | does not make the model fairer; limits where it decides |
| `cue_effect_debiasing` | post-hoc | subtract the audited additive effect of cues detected in the text | in-sample unless effects come from a separate run |

### Every arm is priced

The two easiest ways to pass a fairness audit are to give everyone the same score
and to decline to decide. Both are caught:

- `score_dispersion_ratio` — below 0.5× baseline means the arm flattened scores
  rather than debiasing them;
- `rank_agreement_kendall_tau` — below 0.6 means it changed *who* gets shortlisted;
- `discrimination_retained` — correlation with expected fit must survive;
- `coverage` — abstention is legitimate but must be reported.

An arm is `recommended` only if it reduces the headline bias metrics **and** clears
those guards.

### Why `cue_effect_debiasing` is group-blind

It adjusts for the model's measured sensitivity to specific text cues, never for a
candidate's group membership: demographics are never observed, requested or
inferred at serving time. That distinction is deliberate. Adjusting scores on the
basis of race, colour, religion, sex or national origin is restricted in US
employment testing by Title VII as amended (42 U.S.C. 2000e-2(l)), so a per-group
score adjustment would be legally hazardous even where it improves a parity
metric. Removing a model's measured reaction to a token is a different operation
from re-scoring a person because of their group. This is an engineering note, not
legal advice.

---

## 9. Validating the auditor

Every fairness audit produces numbers. Almost none of them come with an answer to
the obvious question: *does this measurement instrument work?* An audit that
misses a real gap, or invents one, is worse than no audit — it produces a
certificate.

Because the offline provider can inject a bias of exactly known size, the question
is answerable. `frs validate-auditor` runs the **full** pipeline twice:

- **Null condition** (`bias = 0`): the suite should report no significant
  protected-attribute effect. Anything it reports is a false positive, and the
  rate measures **specificity**.
- **Injected condition** (`bias = −3.0` for one level): the suite should recover
  the effect in the right direction, at the right magnitude, and flag it after FDR
  correction. Failure measures **sensitivity**.

The expected magnitude is not guessed. The mock spreads its injected delta across
rubric dimensions with known weights, so the transmitted effect on the overall
score is `bias x BIAS_TRANSMISSION`, a constant computed from those weights.
Current result: expected −2.753, recovered −2.753, CI [−2.83, −2.67], *q* = 0.001.

Crucially, the demographic signals are recovered from the resume **text**, using
the same published lexicons a real model would see — never from metadata handed
over by the pipeline. If the mock were simply told which variant was which, the
oracle would only confirm our own bookkeeping was self-consistent. Reading them
out of the prose puts the whole chain under test.

Two real bugs were found by this oracle and would not have been found otherwise:

1. **The response cache ignored provider behaviour.** The injected condition was
   served the null condition's cached responses, so the recovered effect was
   exactly zero. The cache key now includes a provider `behaviour_key`.
2. **The cache also ignored the provider's *code*.** After a precision fix to the
   signal detector, the oracle still replayed pre-fix responses and the recovered
   effect sat at exactly 0.8× its expected size. For a local provider the source
   *is* an input, so the behaviour key now includes a fingerprint of the modules
   that define it.

---

## 10. Known limitations

- **Sample size.** 12 resumes x 6 jobs in the shipped corpus. The blocked design
  recovers effects far better than a pooled comparison at this size, but small
  gaps remain undetectable, and every null is reported with its MDE.
- **Synthetic corpus.** Real resumes are personal data and cannot be committed.
  Synthetic documents also keep the design balanced — but they may be more uniform
  than real ones, which can make cue effects easier to detect than in the wild.
  `configs/kaggle.yaml` runs the same pipeline over a real public corpus.
- **Cue lexicons encode our model of which signals map to which groups.** They are
  deliberately short and human-readable so a reviewer can disagree with a specific
  entry rather than with an opaque classifier.
- **Effects are model- and prompt-specific.** A cue effect is a property of a
  `(model, prompt, rubric)` triple, not a durable fact. Re-audit when any of them
  changes.
- **`age_marker` is confounded** with career length; its estimate is an upper
  bound that partly reflects seniority expectations.
- **`writing_register` touches far more characters** than a name swap, so its
  magnitude is not comparable to the protected families. It bounds how much of the
  score is style rather than substance.
- **The four-fifths rule is a reporting convention**, not a legal safe harbour.

---

## 11. References

The Phase-I literature review (15 papers) plus the methodological sources this
implementation relies on directly:

- Bertrand, M. & Mullainathan, S. (2004). *Are Emily and Greg More Employable than
  Lakisha and Jamal?* AER 94(4). — the name-perturbation design and the default list.
- Wilson, K. & Caliskan, A. (2024). *Gender, race, and intersectional bias in
  resume screening via language model retrieval.*
- Wilson, K., Sim, M., Gueorguieva, A.-M. & Caliskan, A. (2025). *No thoughts just
  AI: Biased LLM hiring recommendations alter human decision making.*
- Seshadri, P., Chen, H., Singh, S. & Goldfarb-Tarrant, S. (2025). *Small changes,
  large consequences: Analyzing the allocational fairness of LLMs in hiring.*
- Castleman, J., Shen, Z., Metevier, B., Springer, M. & Korolova, A. (2026).
  *Measuring validity in LLM-based resume screening.*
- Cui, L. (2026). *Bias-audited resume screening with calibrated matching scores,
  selective review, and evidence-grounded LLM explanation cards.*
- Rao, P. S. B., Venkatesan, L. N., Cherubini, M. & Jayagopi, D. B. (2025).
  *Invisible filters: Cultural bias in hiring evaluations using LLMs.*
- Tan, B. C. Z. et al. (2026). *Small changes, big impact: Demographic bias in
  LLM-based hiring through subtle sociocultural markers in anonymised resumes.*
- Sivakaminathan, S. S. & Musi, E. (2025). *ChatGPT is a gender bias echo-chamber
  in HR recruitment.*
- Webster, K. T. (2025). *Fairness is not enough: Auditing competence and
  intersectional bias in AI-powered resume screening.*
- An, J., Huang, D., Lin, C. & Tai, M. (2025). *Measuring gender and racial biases
  in large language models.*
- Gao, Z., Jiang, W. & Yan, Y. (2026). *Can LLMs hire fairly? Racial bias in
  resume screening.*
- Hoffstedde, S. A. et al. (2026). *Gender bias in LLM hiring decisions: Evidence
  from a Japanese context.*
- Xu, J., Li, G. & Jiang, J. Y. (2025). *AI self-preferencing in algorithmic hiring.*
- Xu, Z., Yamamoto, S. & Morishima, A. (2026). *AutoScreen-FW.*
- Zhou, B., Jin, J. & Zhao, D. (2025). *Calibrated resume-job matching for
  trustworthy LLM-assisted recruiter screening.*
- Chouldechova, A. (2017). *Fair prediction with disparate impact.*
- Kleinberg, J., Mullainathan, S. & Raghavan, M. (2017). *Inherent trade-offs in
  the fair determination of risk scores.*
- Hardt, M., Price, E. & Srebro, N. (2016). *Equality of opportunity in supervised
  learning.*
- Dwork, C. et al. (2012). *Fairness through awareness.*
- Rivera, L. A. (2012). *Hiring as cultural matching.* ASR 77(6).
- Neumark, D., Burn, I. & Button, P. (2019). *Is it harder for older workers to
  find jobs?* JPE 127(2).
- Rosenman, E., Olivella, S. & Imai, K. (2023). *Race and ethnicity data for
  first, middle, and last names in the United States.* Scientific Data 10:299.
- Monroe, B. L., Colaresi, M. P. & Quinn, K. M. (2008). *Fightin' words.*
- Mitchell, M. et al. (2019). *Model cards for model reporting.*
- Gebru, T. et al. (2021). *Datasheets for datasets.*
- EEOC. *Uniform Guidelines on Employee Selection Procedures*, 29 CFR 1607 (1978).
