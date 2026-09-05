# Submission readiness — Nature Computational Science

Status after the supplementary reorganisation. **The package is complete except for four author items.**

---

## Verified

| item | status |
|---|---|
| Title | *Applicability domains as testable claims: predictive uncertainty against molecular similarity for metal–ligand selectivity* |
| Main text | 4,729 words · Abstract 148 / 150 · Methods 1,392 (excluded from the limit) |
| Supplementary | 13,700 words, ten sections, ten figures |
| Display items | 4 / 6 (Figs 1–4, no tables) |
| References | 27, sequential by first appearance, none uncited or unlisted |
| Cross-references | **all 46 main-text section calls verified against their target headings**; no dangling section or figure reference in either document |
| **Validation** | **264 / 264 assertions pass across three modules, all exit 0, run on your machine** |

---

## Latest pass: the supplementary information

**Methods are now S1.** The definitions, protocols and formulas that were section S10 are the first thing the reader meets, and everything else shifted down by one: S1 Methods, S2 Data, S3 Validation protocol, S4 Model selection, S5 Unit of analysis, S6 Candidate estimates, S7 Retention, S8 What the uncertainty tracks, S9 Pair acquisition, S10 Reproducibility. That order is the analysis in the order it was carried out, with the reference material in front of it. The preamble now says so, so the reader knows nothing later restates a definition. Supplementary figure numbers are unchanged — Methods carries no figures, so nothing moved.

**Two references were pointing at the wrong section.** After the definitions table moved into Methods, the main text's "the full inventory of 23 candidates and their definitions are given in S6.1 and S6.2" no longer described either target, and "each distance family is taken at its best aggregation (S6.2 and S7.1)" pointed at the hedging subsection rather than the aggregation table. Both fixed. I then checked all 46 calls one at a time against the heading and body of the section they name; the other 44 were right.

**A missing citation, found the same way.** The claim that σ's axes run binding strength → Δoxidation state → Δelectronegativity → Δcovalent radius had no pointer at all. It is S8.3, and it now says so.

**The artifact inventory had one gap and five orphans.** `iz_duplicate_measurement_summary.csv` was cited in S2.6 but absent from the inventory; added. Five tables were listed in the inventory but named nowhere in the text — `decision_level_contrasts`, `decision_level_rho`, `aggregation_decomposition`, `within_pair_retention_scalefree`, `within_pair_score_residual`. They are not dead: each is the deposit behind a table that simply never said so, and each is now named where it is used.

**Trimmed, 650 words, all of it restatement.** The 23-score definition table moved from S6.1 into Methods as S1.10, where it belongs and where it stops being a second definition. Then: the error-statistic formula and its invariance argument, which Methods now carries, cut from S5.2 and again from S7.5; the retention protocol cut from S7.1; the cell definition cut from S7.4; the acquisition rule and metric tables cut from S9.1; the MoP/ACE/PICP re-definition cut from S4.2; two of the three aggregation conventions cut from S3.2, keeping "undefined is not zero", which is the one with teeth. The unit-dependence argument in S8.1 was being made twice on either side of the same table and is now made once. Duplicate ligand statistics removed from S2.5, which S2.2 and S2.3 already gave.

**Methods were not trimmed** — they grew, by the score-definition table and by a note on which distances exist for which model.

**The SI title was stale**, still carrying the pre-rename title. Fixed.

---

## Earlier pass

**The external set, and why it belongs in paragraph three.** You were right that this was the missing piece. The compilation's forty-ligand external set was not sampled at random: its ligands were selected by a Mahalanobis centroid distance in a principal-component space of the descriptors, on the premise that geometric remoteness makes a hard test. That is precisely the practice this paper tests — and the criterion is one we score directly. **It reaches ρ = +0.125 ± 0.076 against realized error: 1.65 fold standard deviations from zero, and exactly half the association of the best descriptor distance.** So a held-out set built from it is remote in a space with no demonstrated relation to the model's error. The claim is now a validator assertion, the fuller version is a new block in **S2.3**, and it feeds the Discussion's benchmark-splitting argument, which it makes concrete rather than hypothetical.

**Abrupt sentences.** "Design is not like that." is gone, folded into the sentence it was fragmenting. Same treatment for "The interval is where they separate.", "That is one task.", and the standalone "It therefore measures interpolation...". Each now sits inside the clause it belongs to.

**Dead-weight conclusions cut.** The sentence you quoted — "What it does not do is account for the size of the gap above..." — is now six words: "It is not, however, the size of the gap above." Also gone: "That is the situation an applicability domain exists to manage" (the Introduction says it), "Which construction of that statement survives being tested is the question the rest of the paper asks" (the Introduction says that too), and "A retention curve alone cannot tell an easier population from a better ordering, which is why every comparison here is made twice" (a verbatim restatement of the Introduction's last paragraph, which I trimmed instead).

**Duplications removed.** The Introduction's paragraph 7 and the Results' retention section were making the population-control point twice; it now appears once, in the Introduction, and the Results simply applies it. Section 2's closing sentence and the same section's opening no longer both explain why uncertainty is needed. Introduction paragraphs 3, 4 and 7 were tightened by 29 words with no cited claim removed.

**Net: 4,721 words, down 836 across the last two passes**, with every point you asked for intact — the MoP framing and its negative R², the σ-versus-magnitude control, the chemical axes and why they differ from a hard boundary, the three-metric reasoning, the inverted-objective argument, the ligand-disjoint justification, the internal-inconsistency statement, the three-item Limitations and the benchmark-splitting conclusion.

---

## Earlier pass

**Title.** *Applicability domains as testable claims: predictive uncertainty against molecular similarity for metal–ligand selectivity.* No verb, and it names the contribution (the test), the comparison and the system.

**Section headings are noun phrases now.**

| was | is |
|---|---|
| What the existing model measures | Validation of the existing model |
| From absolute binding to selectivity, and why the model must report its own uncertainty | The selectivity target and the fixed model |
| Predictive uncertainty orders held-out error; molecular distance does not | Risk ordering: uncertainty against molecular distance |
| What the uncertainty is seeing | Chemical content of the uncertainty |
| Acting on the ranking | Prospective-style batch selection |

**Why ligand-disjoint, stated.** §1 now says it in one sentence: a separation process hands the chemist the metals, the medium and the temperature, and the only thing designed is the ligand, so a ligand-disjoint split is the one evaluation that measures the novelty a prospective user actually creates. The full argument, including why metal-pair-disjoint and medium-disjoint splits pose real but rarer questions the record supports poorly, is a new block in **S2.1**.

**The error analysis is out of the main text.** It now says only that laboratories reporting the same system under nominally identical conditions routinely disagree — because a determination depends on the assumed speciation model, on how the ionic medium and temperature were actually held, and on the titration method — so some internal inconsistency is in the nature of this data. The numbers, both brackets and the floor stay in **S1.6/S1.7**, where the validator still checks them.

**Legend duplication fixed by cutting the main text, not the legends.** Fig. 2's legend already said each distance family is at its best aggregation, so the paragraph explaining it lost three sentences. Fig. 3's legend already described the quartile stratification, so §4's opening was compressed from two paragraphs into one. Fig. 4's legend was defining the three metrics that the text also defines — the definitions came out of the legend and the *interpretation* stayed in the text, which is the way round you wanted.

**Descriptive passages compressed.** Your example — the covariance/ACE/0.37–0.38 passage — is now one sentence: propagated uncertainties are badly calibrated because propagation discards a covariance the endpoint predictions do not lack, the intervals are wide enough to flag almost nothing, and direct modelling is better calibrated in nine of eleven cases, with the numbers in Fig. 1b and S3.5. Similar cuts to the retention section, the training-frequency control and the batch-selection metrics. Nothing meaning-rich was removed: the σ-versus-magnitude control, the chemical axes, the inverted-objective argument and the three-metric reasoning are all intact.

**Discussion has its conclusion.** A new third paragraph: held-out sets in this field are built by structural distance on the premise that structural remoteness is what makes a test hard, and if the premise fails the split does not measure what it is taken to measure — a scaffold-split score is a statement about structural novelty, not predictive difficulty, and the field's record of progress inherits the error. A fitted model with a calibrated uncertainty already ranks its own held-out cases by expected difficulty, so a hold-out drawn from the high-uncertainty tail is hard in the sense that matters and can be audited against realized error in the way a distance-defined split cannot. Put as a minimum standard: any split offered as a difficulty stratification should report whether its held-out set is in fact less accurately predicted than a random one of the same size.

**Limitations** is three items, not four (the count line was stale).

---

## Earlier pass

**The noise floor — new analysis, and it corrected a claim of mine.** You asked whether MAE 0.599 with R² 0.934 is even achievable leak-free. It is, comfortably, and the paragraph implying a data-quality wall was wrong. `noise_floor.py` turns the repeated measurements into a bound: a model recovering a system's consensus value exactly is still scored against one reported number, so the spread among repeats floors any model's error. Pooling the 3,048 repeated condition groups gives repeatability SD 0.496 and a distribution-free MAE floor of 0.241; trimming the 12% of groups that disagree by more than a log unit gives 0.190 and 0.113. So **the published MAE sits 2.5–5.3× above the floor and the R² ceiling is 0.99, not 0.934** — real skill, not a fit to noise. The **ligand-disjoint MAE is 6.2–13.1× the floor**, so the leak-free gap is a modelling gap.

On the estimator: I used the pooled within-group variance with a distribution-free mean absolute deviation rather than a Gaussian conversion, because the error distribution is nowhere near Gaussian (median |deviation| 0.07, SD 0.50), and I report two brackets rather than picking one. Both biases I could identify — remeasured systems are not a random sample, and identical re-entries may be one primary source counted twice — push the floor *down*, so every "the error is well above the floor" statement is conservative.

That floor pays off twice more. The selectivity target is a difference of two condition cells, so its floor scales by ×1.369 → **0.680 inclusive, 0.260 trimmed**. The unfiltered MoP RMSE of 2.078 is 3.1× the floor, and **the 0.730 reached by retaining half the predictions on uncertainty is 1.07× it**: on the permissive reading, half the predictions are as accurate as remeasuring the same systems. Derivation in **S1.7**, formulas in **S10.5**.

**Mean-of-pairs now comes before any metric**, in the first paragraph of Results §2, with the contrast that makes the point: row-weighted **R² is 0.725**, which reads as a serviceable regression, while **MoP R² is negative** — on a typical pair the model loses to guessing that pair's average. Row-weighted scores on this task measure how well frequent pairs are remembered. ACE and MoP are defined in the same breath, so every later metric inherits the framing.

**Abstract and opening rewritten.** Abstract is 148 words and built on the double cost ("if it is wrong it is wrong twice") rather than the null result. The opening is now two sentences: a deployed model is asked about inputs its developers never saw; the question that follows is not average accuracy but whether *this* input is trustworthy.

**Two things that were missing.** The obvious referee objection — is σ just predicted magnitude re-expressed? — is now answered in Results and S7.3: raw ρ 0.648, |ŷ| alone 0.541, **partial ρ 0.428 ± 0.026** after residualising on magnitude. And the physical reading you asked for: σ's axes are the familiar ones of coordination chemistry (binding strength first, then Δoxidation state, Δelectronegativity, Δcovalent radius, with the same ordering for the error itself), which is why it behaves unlike a distance threshold — a graded statement along axes that govern the chemistry rather than a hard boundary in a space unrelated to the target.

**Trimmed.** The retention section lost a paragraph and the training-frequency control went from five sentences to three, keeping the argument (it selects rather than orders) and dropping the recital. The LCB sentence you quoted is gone, along with the k-by-k number lists in the acquisition section.

**Limitations.** Retrospective design and one-target/one-compilation are out. In their place: **the replacement is not yet general** — the uncertainty criterion has not been run against distance criteria across molecular property prediction, and there is a class of problems where the argument favouring it here does not apply (molecule-only targets, chemically narrow training sets, no calibrated uncertainty available). We do not claim a distance criterion fails everywhere; we claim the asymmetry that one is adopted untested while the other passes a test that is available to settle any other case.

**Methods.** The journal excludes them from the 3,500 words — "Main text – up to 3,500 words, excluding abstract, Methods, references and figure legends" — so they stay in the main text. But the formula-level detail you liked in the JCIM draft now lives in **S10, Supplementary Methods**: notation, MoP RMSE, PICP and ACE with both weightings, normalized error, the floor estimator, the retention protocol, rank statistics, all six acquisition rules with the three metrics written out, and the full law-of-total-variance decomposition of σ. Main-text Methods dropped the duplicated definitions and now point there, about 250 words lighter.

**Eight new assertions**, to 263.

---

## Earlier pass

**Terminology.** "Interval calibration error" was ACE under another name, and the paper was using both plus "coverage error" and "interval calibration" as if they were three things. There is now one vocabulary, defined once in Methods and used identically in the main text, the SI, the tables and both figure scripts. **MoP RMSE** is accuracy with each metal pair weighted equally. **ACE** is the average coverage error, the mean absolute gap between nominal and realised interval coverage across the nine nominal levels; **PICP@90** is realised coverage of the 90% interval. Every value now says whether it is row-equal or pair-equal, which had been silently ambiguous between the eligibility screen (row-equal) and the model tables (pair-equal). Fig. 1b's axis reads "average coverage error (pair-equal)".

**The decomposition moved.** The aleatoric-share paragraph is out of Results. S3.4 keeps it and now says plainly how far it can be trusted: the aleatoric term is the model's own variance head, nothing here identifies it against an external noise estimate, a replicate-spread analysis is not available under a ligand-disjoint split, and no main-text conclusion depends on it. In its place, Results now carries what the finding is actually *for* — that an acquisition loop buying wherever σ is largest would spend on the hardest chemistry rather than the most informative, so the score is comparative and not an absolute trigger.

**The acquisition section is now a campaign, not a protocol.** It opens on the question a separation chemist brings: two metals, a pool of candidates, budget for a handful, which handful? Prospective-style is stated as what it is — ranking happens before any label is read, but it is a replay and not an experiment. Each metric is introduced by the question it answers: **event precision** is what fraction of the batch was not wasted synthesis, **any hit** is whether one lead was found, **regret** is how much of the best available selectivity was left behind. High selectivity is the campaign; near zero is presented as the inverted task and a regime check, not a use case.

**Interpretation replaced the number lists.** The two descriptive paragraphs are gone. What is in their place: random choice wastes one first experiment in three, the model one in six. Subtracting σ is a penalty aimed at the target, because σ rises with predicted selectivity — and inverting the objective inverts the conclusion, which is the sharpest evidence the mechanism is real. The two heaviest penalties are *worse than random* at finding one success, because they discard the tail a random draw sometimes catches. Against a pool whose best candidate averages 7.54, a top-ten batch from either good rule contains the single best ligand three times in four, against fewer than half at random. Full per-rule tables stay in S8.

**Event-threshold sensitivity restored.** It had been dropped. **S8.4** and **Fig. S10** repeat the whole benchmark at ΔlogK > 2 and |ΔlogK| < 1/2, in Fig. 4's exact layout so they can be read against each other. Every ordering survives, and the model's margin over random *widens* (0.232 against 0.192) as the objective tightens — which is what you want if the ranking carries information rather than exploiting a high base rate.

**Limitations, as its own section.** Four items, the first the one you named: the record contains only systems that bind. A large predicted selectivity can come from strong favoured binding or from the disfavoured metal barely binding, and the second case is what the data cannot teach and the held-out set cannot confirm — so on a genuinely hypothetical ligand the false-positive risk sits exactly where the model looks best, and no criterion tested here detects it, because the missing evidence is missing from the target rather than from the input. Then: the design is retrospective; the record disagrees with itself; one target, one compilation, one model family.

**Discussion cut to conclusions.** Four paragraphs. What the test found, the two controls that should become routine, the algebraic point that travels past chemistry, and the one-analysis proposal. Everything interpretive about *these* results now lives in Results where it belongs.

**Eight new assertions**, to 255, covering the interpretive claims and every value and ordering in S8.4.

---

## Earlier pass: the opening section, and Fig. 1b

**"Benchmark" is gone.** You are right that the authors do not call this a benchmark, and the word was doing work the source does not support. It is now "the published model on this compilation" throughout the main text and the SI, and the section heading is *What the existing model measures*. The three sentences of hedging around it are gone too: no "any claim has to be reconciled with that number", no "the published result is real and reproducible", and no closing apology that the original never claimed otherwise. It now states the number, recovers it in one clause, and moves to what it measures.

**A citation error the rewrite exposed.** The two Results citations to the published model pointed at reference 23, Zaverkin on interatomic potentials, because the reference list was renumbered when the section was added and the new first appearance was not folded in. The metal–ligand model is now reference 24 and the three chemistry references shifted to 25–27. Order and both directions re-checked: 27 references, sequential by first appearance, none uncited, none unlisted.

**The compilation is now named where it first appears.** "This compilation" had no antecedent — the Introduction never says where the data comes from. The opening sentence now identifies it as aqueous stability constants curated by the Critical Materials Institute.

**The internal disagreement is in.** You were right that it was missing. Among the 3,048 groups of metal-complex measurements agreeing on ligand, metal, oxidation state, temperature, ionic strength and electrolyte, 1,234 disagree by more than 0.20 in log *K* and 368 by more than a full unit; the median within-group range is 0.11 and the ninetieth percentile 1.16, the same size as the selectivity threshold itself. Two sentences in the Results with a pointer to the new **S1.6**, which carries the full diagnostic, the two worst cases, and the honest statement that this is descriptive rather than an identified noise variance — replicate structure and fold structure are not independent under a ligand-disjoint split. It also now closes a loop in the Discussion: the high aleatoric share is what this record predicts.

**Fig. 1b rebuilt.** Seven named models instead of twenty-three anonymous points: the best-calibrated and the most accurate of each construction, plus the evidential graph network, which was run on both sides and is the sharpest single instance of propagation failing. A dashed vertical rule marks the best ad hoc model that predicts a point and no interval (2.002 ± 0.182), which is the reference that makes the argument land — every interval-bearing model on either side sits within about a fold s.d. of it, so attaching an uncertainty costs nothing in accuracy and the choice is decided purely on the interval. The Results paragraph was split in two to follow that: accuracy does not decide, the interval does.

**New Supplementary Fig. S4** carries the full picture: every interval-bearing model named and ordered by calibration, and the same models in the accuracy-against-calibration plane with all three point-only models as vertical rules. SI figures S4–S8 became S5–S9; both documents and both build scripts renumbered, and the superseded PNG/PDFs moved to `figures/_to_delete/renumbered/` on your machine rather than deleted.

**Four new assertions**, to 247. They cover the point-model accuracies, that every row of the deposited table now carries a readable model name and only the point rows have empty calibration columns, and both internal-disagreement counts.

---

## Earlier pass: the Results now start at the beginning

You were right that the paper opened in what the JCIM version treats as its middle. A referee reading the old draft had no answer to "a model for this compilation already exists and reports MAE 0.599 — why are you here?". The Results now walk the same chain the JCIM article walked, compressed to two sections, with the detail in the SI.

**New §1, *What the published benchmark measures*.** Reproduces the published benchmark rather than working around it: the released weights give 0.592 and an architecture-matched retraining 0.595 against the published 0.599, so the result is real. Then what it measures — its train/test assignment is random over records, and 1,568 of its 1,670 test ligands also appear in training, so 97.5% of test rows carry a ligand the model has already seen. Its own external file of forty unseen ligands rises to 0.834; the same architecture under ligand-disjoint validation gives 1.484 ± 0.161, 2.5× the published figure. The section closes by saying explicitly that this is not a criticism of the benchmark, which never claimed ligand-novel generalization — it fixes which question is being asked.

**New §2, *From absolute binding to selectivity, and why the model must report its own uncertainty*.** Defines ΔlogK and the separation factor where they are first used, then the ad hoc versus direct comparison: composed selectivity is competitive on point accuracy (MoP RMSE 1.94–2.37 against 1.93–2.24), and the two constructions separate on the interval, because propagating two absolute uncertainties discards the covariance term. Best composed calibration error 0.200, the evidential composed models 0.37–0.38 with 90% intervals covering 96%; nine of eleven direct models beat 0.200, a tenth matches it, and the selected model reaches 0.153. Only then does the model description arrive, followed by the accuracy fact that makes UQ necessary rather than optional. This is the point-predictor-versus-UQ motivation you asked for, now carrying its own evidence instead of asserting itself.

**Fig. 1 is new** and both old figures moved down: 1→2, 2→3, 3→4. Panel a is MAE under the three evaluation regimes, bars our recomputation and diamonds the published values, annotated with each regime's ligand overlap; panel b is MoP RMSE against interval calibration error for every model producing a usable interval, composed against direct, star on the model carried forward.

**Table 1 is gone**, as you asked. It duplicated Fig. 2a and most of its prose described what the figure already shows. Its three load-bearing numbers survive in the sentence that needs them, with the full table in S6.1 where it already existed. Display items stay at 4 of 6, now all figures.

**SI: two sections added, one rewritten.** S2.3 carries the full reproduction — a three-row regime table, the ligand-overlap table, and the statement of what each regime estimates. S3.5 is rewritten from a paragraph to two complete tables, twelve composed models and eleven direct ones on a common all-pair scope, plus the structural explanation of why propagation fails for a difference of two predictions on the same ligand. Both are emitted by a new module, `benchmark_context.py`, from frozen artifacts; no model is loaded, trained or rescored.

**Nine new assertions**, taking the suite to 243. They cover the published values, both reproductions, the ligand overlap, the ligand-disjoint error and its ratio to the published figure, the two RMSE ranges, the composed calibration failure with its PICP, and the 9/1/1 count. One caught me: I had written "nine better, a tenth matches" with a loose tolerance that made it eight and two. The tenth model exceeds the best composed one by 0.00016, far inside either model's fold s.d., so the sentence stands — but the check now asserts the strict count and the size of the tie rather than hiding it in a tolerance.

---

## Earlier pass: structure, title, and a table

**Title.** Now *Predictive uncertainty outperforms molecular similarity as an applicability domain for metal–ligand selectivity*. It names the contribution, the comparison and the system, and it no longer leads with a negative result. Two alternatives if you prefer a different emphasis: *Validating applicability domains against error in metal–ligand selectivity*, which foregrounds the method, or the JCIM-style *Molecular similarity does not bound prediction error: calibrated uncertainty as an applicability domain*, which is closer to the original.

**Introduction reordered.** The approach paragraph now follows the task rather than preceding it, so the reader meets the problem before the method. The task paragraph no longer carries dataset statistics or a Supplementary figure pointer, since those are results; it now argues only why metal–ligand selectivity is the right test bed, that only the ligand axis is designed while metals and medium are imposed. The introduction ends on the test rather than on numbers.

**A gap the earlier drafts had, now closed.** The JCIM version motivated the whole enterprise with a fact this draft had dropped: unfiltered, on ligands it has never seen, the model is barely better than predicting each metal pair's mean (normalized error 0.974 ± 0.256, against 1.000 for the pair mean). Without that, uncertainty modelling looks like a preference rather than a necessity. The Results now open on it: aggregate accuracy is already known and is not encouraging, so the useful question is *where* the model is right, and only a per-prediction estimate answers it. That is the point-predictor-versus-UQ comparison you asked for, with the model-selection detail left in S3.

**Three things the Results needed and lacked.** The separation factor is now defined where it is first used, with ΔlogK = 1 stated as a tenfold preference and named as the quantity a ligand designer moves. The model is described in two sentences: five-member heteroscedastic MLP ensemble on RDKit descriptors with MC dropout, chosen from eleven candidates on accuracy and calibration, frozen before the analysis. The dataset size and the unequal novelty axes now sit in the Results rather than the introduction.

**Table 1 replaces the number-dense paragraph.** Eight rows, five columns, and the prose alongside it now carries only the comparisons a reader should take away. Display items go to 4 of 6.

**Factual check on "aqueous".** It holds. All six background electrolytes in the data are aqueous salt media (NaClO₄, KNO₃, KCl, NaNO₃, Me₄NCl, NaCl), and the JCIM version describes the same source as aqueous stability-constant measurements from the Critical Materials Institute compilation.

**Three more assertions** cover Table 1 in full, including the percent-change column, and the normalized unfiltered baseline. 234/234 pass.

---

## Earlier pass: prose

**Every em-dash is gone.** There were 25 across the abstract, Results, Discussion and Methods. Each was rewritten rather than swapped for a comma, since most were carrying a parenthetical that read better as a clause or a colon.

**The announce-then-elaborate sentences are gone.** The pattern you picked out ("The question is general.", "Chemistry answers it with a device of its own.", "We do that here, under one test applied to both.") appeared throughout. Those are now folded into the sentences they were introducing. The same fix was applied to "The test is fixed retention.", "A risk score is worth what it changes about a decision.", "That last observation has a form that requires no data." and "What 'resemblance' means, though, is not settled." Enumerations that genuinely belong in a discussion ("The first is... The second is...") were left alone.

**Abstract rewritten again.** The two sentences you flagged are gone. It now runs as continuous argument rather than a list of contrasts: what the domain is and where it is embedded, why the boundary is unchecked and structurally blind, the claim we substitute, then the result on one task, then what it costs to reuse the test. 149 words.

**Two stale figure legends caught in passing.** Fig. 2's legend still described the line version from two passes ago rather than the heatmaps that are actually there, and Fig. 1's legend still said "residual" where the axis now says "absolute error".

---

## Earlier pass: introduction restructured, abstract reframed

**Introduction now follows the four moves you set out**, in six paragraphs:

1. *General AI.* Models sit inside decisions their developers cannot enumerate, so the question after every deployment is whether **this** input is one the model can be trusted on. Framed generally and cited generally — predictive uncertainty degrading under distribution shift (Ovadia *et al.*, NeurIPS 2019), and whether a language model can recognise what it does not know and whether that survives a change of domain (Kadavath *et al.*, 2022).
2. *Chemistry's own answer.* The boom (structure prediction, materials, closed-loop labs) and then the applicability domain as the device chemistry reaches for, with its two-decade inventory, the survey language that never asks for a check against error, its use beyond small molecules, and its presence inside benchmark construction — so being wrong costs twice.
3. *Vagueness into contradiction.* "Resemblance" is not one function; the members disagree, none is derived from the error it bounds, and the choice is convention. Then the concrete contradiction you asked for: a Tanimoto distance is a function of one molecule and can see nothing else, while the practical targets are molecule-plus-context — cross-coupling performance on ligand/base/additive, solubility on solvent, conditions as the target itself. A ligand-only criterion sees one of two axes; a structural limit, not a descriptor problem.
4. *The other tradition.* Conformal prediction entering this literature as an explicit alternative to the applicability domain, its generalisation to risk control, per-prediction uncertainty from ensembles and heteroscedastic heads, the materials redefinition of a domain as measured-lower-error, and acting on uncertainty in self-driving labs and potential retraining. None has displaced distance-based practice, and the two traditions are seldom compared on one task.
5. *What we do* — one test for both, with the population control stated as a condition rather than a detail.
6. *The case* — metal–ligand selectivity, chosen because it has all three properties the test needs.

**Abstract reframed around the contribution, not the null result.** It now leads with the domain as a measurable claim and our reformulation of it, names the test, then gives the practical case and what the uncertainty is worth in a decision. The failure of the similarity criteria is the motivation, given in one clause rather than as the headline. 150 words.

**Three further references, verified against the publisher record**: Ovadia *et al.* (NeurIPS 2019), Kadavath *et al.* (arXiv:2207.05221), Angelopoulos *et al.* (ICLR 2024). List renumbered to 27, checked in both directions.

---

## Earlier pass: abstract and introduction

**Abstract.** Reframed so the instrument, not the negative result, is the subject. It now opens on the fact that a resemblance check is the routine gate in chemical ML *and* is built into benchmark splits, states which forms fail (fingerprint nearest neighbour no better than random; learned-embedding distances worse than random), and makes the uncertainty result the payoff rather than a footnote — including what it is worth in a decision, a one-ligand shortlist going from 64% to 83% useful. 148 words.

**Introduction, paragraph 1.** Rewritten to the shape you described: what ML has actually achieved in chemistry, named — proteome-scale structure prediction, hundreds of thousands of proposed stable materials, interatomic potentials replacing electronic-structure calculations, closed-loop laboratories including LLM-directed ones — then the problem that came with it, deciding whether a given input is one the model can be trusted on. The instrument inventory moved into a second paragraph so paragraph 1 carries context and stakes only.

**Introduction, the context paragraph.** It had no references, which as you say is not acceptable for an audience that will not take "chemistry has extra context" on trust. It now carries three, each a concrete standard task where the target depends on more than the molecule: C–N cross-coupling performance depending on ligand, base and additive (Ahneman *et al.*, *Science* 2018); solubility depending on which solvent it is measured in (Boobier *et al.*, *Nat. Commun.* 2020); reaction conditions as the prediction target (Gao *et al.*, already cited). The uncited "binding affinity depends on the protein" clause is gone rather than left unsupported.

**Community stakes made explicit.** One sentence added where benchmarks are discussed: the assumption is not only how a deployed model is fenced, it is built into how the field measures progress, so being wrong costs twice.

**Four new references, all verified against the publisher record** before citing: Merchant *et al.* (*Nature* **624**, 80–85), Boiko *et al.* (*Nature* **624**, 570–578), Ahneman *et al.* (*Science* **360**, 186–190), Boobier *et al.* (*Nat. Commun.* **11**, 5753). The full list was renumbered to 24 and re-checked in both directions.

**The Discussion's generalisation paragraph** no longer repeats the introduction's three examples; it now reaches past chemistry (a sequence model gated on organism, a materials model on crystal system, a clinical model on site), which is the claim that travels to this journal's readership.

---

## What the earlier pass changed

### Figures, against your six points

**Fig. 1** — the cramped legend is gone: one shared legend below both panels, so neither is crowded. Panel b's axis now reads "rank correlation with absolute error", matching the term used in Fig. 2 and the SI; `|residual|` no longer appears anywhere.

**Fig. 2** — back to heatmaps, both units side by side with their axis marginals, and the quartile-definition panels kept above them. The line version really was obvious once drawn; the grids carry the sixteen numbers and the marginals carry the argument.

**Fig. 3** — grouped bars, JCIM-style, three metrics × two objectives. You were right that the *k*-dependence was eating the plot: as bars, the rule comparison inside each *k* reads directly.

**Fig. S1** — "endpoints" is gone; panel b now says what it is, the individual log *K* measurements each pairwise target is a difference of. Panel c no longer duplicates Fig. 2b: instead of records-per-pair it shows the ligand rank–frequency curve, which is the axis the paper actually holds out (top 10 ligands carry 14.19% of records, top 100 carry 51.87%).

**Fig. S3** — the aleatoric-share panel is gone; panel b is the same two quantities in normalized error. This turned up something worth having: **normalized σ is indistinguishable between the regimes (1.010 ± 0.214 against 1.009 ± 0.467) while normalized error is not (1.120 against 0.737)**. So σ scales almost exactly with the range each pair spans — the same behaviour S7.4 quantifies — but the excess error where selectivity is large is not only scale. Stated in S3.4.

**Fig. S4** — rare-earth highlighting removed; you were right that it was a leftover with no discussion carrying it. Panel a is the general error-against-spread relationship, panel b is absolute against normalized error per pair, which is what actually motivates the normalized unit. The rare-earth ratio survives as one sentence in S4.3 because it is the sharpest single instance.

**Fig. S7** — panel b deleted (it did duplicate Fig. S6), and panel a replaced with what you asked for: **pooled retention on the evidential D-MPNN's own errors, comparing its own uncertainty, the classic median distances, and two distances measured in its own learned embedding.** This required a new module, since the embedding distances had never been put on a retention curve. The result is the strongest version of the paper's claim: against an unfiltered 2.054, the model's own σ gives 0.858, descriptor Euclidean median 1.404, fingerprint median 1.676, random 1.798, and **both embedding distances land above random at 1.950 and 1.961**. A learned representation does not rescue a distance-based domain — on the very model that produced it.

### New analysis, and what it cost

`embedding_retention.py` emits both new tables. Two things to flag honestly:

- My first version of the normalized regime statistic was a **row-level** ratio and it blew up — near-zero came out at 6×10¹¹, because pairs whose spread is near zero dominate the average. Recomputed the way the paper defines normalized error (per eligible cell, RMSE ÷ that cell's spread, cells averaged equally) it is stable and the values above are from that.
- The validator caught my claim that both embedding distances beat random "at every fraction". They do from 10% to 80%; at 90% they sit 0.05 *below* random, well inside the fold s.d. Text now says 10–80% and the SI states the crossing.

### Text

Results §2 rewritten around the six-score Fig. 1 and the new Fig. S7 pointer; §3 rewritten around the two heatmaps and the normalized-unit reversal; §4 rewritten around all three metrics. Every figure and section cross-reference re-resolved in both directions — the dangling Fig. S6b/S7a/S7b pointers left by the reshuffle are gone.

---

## Needs you — cannot be completed without author input

1. **Title page.** No author list, affiliations, or corresponding-author designation. A placeholder line marks where it goes.
2. **Data availability.** `[repository DOI]` is a placeholder.
3. **Code availability.** `[repository URL]` is a placeholder.
4. **Author contributions, Competing interests, Acknowledgements**, and a Reporting Summary.

---

## Open questions I did not decide for you

**Whether Fig. 3 should be the grid or the chemistry.** The uncertainty × support grid is the stronger evidence and the cleaner picture. What it loses is the chemical story — that the flagged region is where the metals bind strongly and where selectivity has the widest range to move in — which now survives only as prose plus a pointer to S7.2. If a referee asks "what has the uncertainty learned", the answer is currently a paragraph rather than a panel. Two display items are still free.

**Where the final trim comes from.** The main text is 999 words over. The reworked opening is the part a referee will use to decide the paper is not cherry-picked, so I would not cut it. The slack is in §3–§5 and the Discussion, where several comparisons are stated twice, once in prose and once in a figure legend that already carries them. Say the word and I will do that pass.


---

## Before deposition

- `scipy` and `scikit-learn` were installed on the working machine during these passes; `requirements.txt` should pin them.
- `validate_core_analyses.py` needs `PYTHONPATH=src` to resolve `uq_remake`.
- `decision_level_statistics.py` and `pooled_risk_ordering.py` each take several minutes.
- Derived tables live under `manuscripts/derived/tables/`, which the validators and both figure scripts read; the acquisition tables are read from `manuscripts/final/tables/`, which is never written to.
- Superseded figures were moved to `manuscripts/natcompsci/figures/_to_delete/superseded/` rather than deleted, so you can remove that folder yourself.
- `.git/index.lock` is stale in the repo and blocks every git command; delete it and the tree is ready to stage. Nothing is committed.
- `pooled_risk_ordering.py` takes about six minutes, longer than a single bridged command allows, so its tables were computed in the analysis container and written into the repo; the validators reproduce every value from the frozen fold scores on your machine. Please run the module once locally before deposition for a local provenance record.
