# Referee report — simulated

**Manuscript:** *Similarity-based applicability domains fail at the level of the decision they gate*
**Journal:** Nature Computational Science — Article
**Reviewer perspective:** computational-science referee; read main text and figures closely, SI where the main text points.

---

## Editorial assessment

**Send out for review.** This is not a desk reject. The framing is genuinely computational-science-facing — the opening establishes reliability estimation as a cross-cutting problem with instances in structure prediction, interatomic potentials, sequence modelling and autonomous experimentation before narrowing to a testbed — and the central methodological argument is not domain-specific. That is a real change from the version this appears to have been reshaped from.

**Recommendation after review: major revision.** The work is careful and the reporting standard is well above average, but three issues need addressing before I would support publication, and one of them is structural rather than cosmetic.

---

## Summary

The authors fix an uncertainty-bearing regression model under ligand-disjoint cross-validation, freeze it, and then test whether the reliability estimates in routine use — fingerprint and descriptor distances, a training-frequency count, and the model's own predictive uncertainty — actually order its held-out error. The test is posed at the level at which such an estimate is consumed: ranking candidates within a single decision. They report that one family of criteria is undefined at that level by construction; that fingerprint distances are negatively associated with error and perform no better than random discarding; that predictive uncertainty halves the retained error; and that training frequency is a confounded proxy for the underlying physics.

---

## Major points

### 1. The flagship argument has no documented instance, and without one it reads as a tautology

The paper's most transferable claim is that an estimate computed on the context of a prediction is constant across the candidates in a decision and therefore carries no information about it. I agree, and I agree it requires no experiment. That is precisely the problem: as stated, it is a statement about constants, and a sceptical reader will ask whether anyone actually does the thing being criticised.

The manuscript never shows them doing it. It asserts that "thin representation is routinely taken to imply restricted applicability" (Introduction) and that coverage-based domains are of the undefined family, but it does not point to a single published analysis that validates a frequency- or coverage-based criterion on pooled data and then applies it to a decision. Without that, Result 2 is either a tautology or a strawman, and a reviewer less sympathetic than me will say so.

This is fixable and the material appears to exist — the manuscript cites a benchmark (ref 9) whose authors attribute elevated error to underrepresentation of a class of inputs and infer restricted applicability from it. That is exactly the inference at issue, made in print, on this dataset. One short paragraph documenting a real instance, with the quotation, would convert the strongest general claim in the paper from an assertion about algebra into a diagnosis of a practice. I would insist on it.

### 2. The analysis excludes the regime in which an applicability domain is actually invoked

Cells require at least twenty held-out ligands (Methods; Supplementary S4.1), which restricts the analysis to well-measured metal pairs. The authors note this in a single clause of the Discussion. It deserves considerably more, because it cuts at the central claim.

An applicability domain is not usually invoked where data are plentiful. It is invoked at the sparse edge — the unusual system, the thinly measured context — which is exactly the population the eligibility rule removes. A reader can reasonably respond: *you have shown that similarity-based estimates add nothing where the model already has support; that is not where anyone claims they help.*

I do not think this is fatal, and the authors have a partial answer they have not made: because the split is ligand-disjoint, every test ligand is unseen regardless of how well its metal pair is covered, so ligand-level novelty is present throughout. What is untested is the *joint* sparse regime — rare pair and novel ligand together. The manuscript should say this explicitly rather than leaving the reader to construct the defence, and should report a sensitivity analysis at lower cell thresholds (n ≥ 10, n ≥ 5) showing whether the ordering of scores is stable as the sparse tail is admitted. If the conclusions hold as the threshold drops, the paper is considerably stronger; if they change, that is itself a result worth reporting.

### 3. Whether the conclusions are conditional on model skill is unaddressed

The manuscript states plainly that the model's error equals the within-pair spread, i.e. it resolves a few per cent of within-pair variance. I appreciate the honesty, and the authors pre-empt the obvious objection well: because predictive uncertainty *does* order this error at ρ = 0.436, the error is demonstrably predictable, so the similarity measures fail against an attainable target rather than against noise. That is a good argument and it should be kept.

It does not, however, address a different question. Similarity-based domains are premised on the idea that error concentrates where the model is extrapolating structurally. If a model has learned structure–property relations well, that premise might hold; if it has learned them poorly, its errors may be governed by something else entirely — as the authors' own analysis suggests, by interaction strength and measurement scatter. The reported failure of similarity may therefore be a property of models in this skill regime rather than of similarity measures as such.

The Limitations do not mention this. They should, and the manuscript would be much strengthened by evaluating one further model of substantially different skill — the grid of eleven candidates described in Supplementary S3 presumably contains one — and reporting whether the sign and ordering of the score–error associations are preserved. This is the single most valuable additional analysis available from data already in hand.

### 4. One system

One target property, one compilation, one model family. The authors are careful to separate the definitional claim (which generalises) from the empirical results (which do not), and I credit that. But a paper whose title makes an unqualified claim about similarity-based domains rests that claim on a single dataset. Either a second system, or a title and abstract calibrated to what one system can support.

---

## Minor points

- **Internal inconsistencies.** The abstract is labelled 148 words and is 139. The header states a main text of 2,906 words; it is 3,043. The Methods reproducibility statement describes 22 + 56 = 78 assertions, while Supplementary S8.1 describes 112 across three modules. Individually trivial; collectively they undermine confidence in a manuscript whose principal virtue is numerical care. Please reconcile.

- **Figure 1 carries no result.** Panel (a) is data description and panel (b) supports a limitation stated in the Discussion. Neither is a finding. With three of six permitted display items used, this is not a space problem, but the first figure of an Article setting out data characteristics rather than results is a weak opening. Consider moving it to Supplementary and promoting the decision-level comparison to Fig. 1.

- **No view of the model's predictions.** The manuscript reports only second-order statistics. A parity plot or held-out error distribution, even in Supplementary, would let a reader confirm the model behaves sensibly before accepting conclusions drawn from its error structure.

- **"Halving."** A reduction from 0.974 to 0.516 is 47%, described twice as halving. Minor, but in a manuscript this precise it stands out.

- **Random baseline spread.** At r = 0.50 the random baseline is 0.928 ± 0.186; the fold s.d. exceeds the gap to every similarity score. The authors' claim — that fingerprint distance is not distinguishable from random — is correctly stated and correctly hedged in Supplementary S6. Please ensure the main text does not read as claiming fingerprint distance is measurably *worse*.

- **Title.** Accurate but framed as a negative result. Nature Computational Science readers respond better to a claim about what to do. Consider foregrounding the prescription (validate at the level of use) rather than the failure.

- **Supplementary S8.3** — reporting the analyses that were run and not included, with reasons, is exemplary and I would like to see it become standard. It also pre-empts the most common reviewer suspicion. I mention it because it deserves saying, not because it needs changing.

---

## What is strong

The methodological discipline is unusual and should be preserved through revision. One unit of analysis and one error statistic are declared before any result and held throughout, with the scale-invariance of the choice demonstrated rather than asserted. The model is frozen before the analysis is designed, and the manuscript says so in terms that can be checked. The predicted-magnitude comparator is an elegant control: showing that ranking by the prediction alone is *worse than not filtering* simultaneously refutes the "uncertainty is a relabelled prediction" objection and establishes that the error is orderable in principle. The coverage-by-uncertainty interaction is a genuine and useful refinement of a widely held intuition, and the authors resist overstating it. The reproducibility apparatus is better than almost anything I review.

---

## What would change my recommendation to accept

1. Document a real instance of the practice the definitional argument targets (Major 1) — one paragraph.
2. Report the cell-threshold sensitivity analysis and state the joint-sparse limitation explicitly (Major 2).
3. Test one further model of different skill and report whether the ordering holds (Major 3).
4. Reconcile the internal inconsistencies and address the minor points.

Items 1, 2 and 4 require no new computation beyond what the deposited artifacts support. Item 3 requires re-running the existing analysis against a second frozen model, which the described pipeline appears to make straightforward. A second dataset (Major 4) would make this a strong paper rather than a good one, but I would not hold publication for it provided the title and abstract are calibrated accordingly.
