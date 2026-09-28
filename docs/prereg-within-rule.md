# Within-rule analysis: written down before it is computed

The full matrix is done for four models. Looking back at the evidence-swap
conditions, most of the swapped-in captures don't contain anything the alert's
rule fires on (the published detector fires on the donor in 19 of 80
cross-label swaps and 25 of 80 same-label swaps). A model that answers "this
alert isn't supported" there is doing what an analyst would do, so those
conditions measured whether the alert is supported, not whether the model tells
attack from benign activity.

The obvious fix, swapping only donors the rule fires on, turns out to cost
nothing: once the match count left the alert, the alert carries only the rule's
title, id and severity and an opaque capture handle. "Rule R's alert with capture
D's events" is then exactly the reference case R@D, which already ran three
times. So the question can be asked of the reference runs that exist.

This file fixes how, before the numbers are looked at. The aggregate reference
results for every model have been seen; the numbers below, on this subset, have
not.

## The subset

Rules the published detector (detection-under-load at `3a2502e`,
`benchmark/results.json`) fires on for at least one true-positive capture and at
least one false-positive capture. For these rules the alert text is the same
whichever capture it came from, so any difference in verdict between their
captures has to come from the telemetry. Six rules, 50 cases (21 true
positives, 29 false positives) on 21 captures:

| rule | cases | true positives |
|---|---|---|
| `250ae82f-736e-4844-a68b-0b5e8cc887da` | 8 | 4 |
| `4a1b6da0-d94f-4fc3-98fc-2d9cb9e5ee76` | 13 | 3 |
| `4b447e9d-1c82-47f6-9a01-a1bb0a22d684` | 12 | 7 |
| `5ef9853e-4d0e-4a70-846f-a9ca37d876da` | 8 | 4 |
| `678dfc63-fefb-47a5-a04c-26bcf8cc9f65` | 6 | 2 |
| `962fe167-e48d-4fd6-9974-11e5b9a5d6d1` | 3 | 1 |

`score/within_rule.py` derives this list from `results.json` rather than from
this table, and refuses to run if the two disagree.

## The one comparison that decides it

Model: gpt-6-luna, picked because it scored highest on the full reference run.
That is a choice made after seeing results, and the claim below is about Luna on
this corpus, not about which model is best.

Luna's reference run against Luna's forced guess from the alert alone
(`alert-only-forced`), on the 50 cases. A case's answer is the majority over its
three repeats; without a majority it counts as unanswered. Score is macro-F1.
The difference (reference minus forced) gets a 95% interval from 2,000
resamples of captures, true- and false-positive captures resampled separately,
seed 0, as in `score/analysis.py`.

Luna is called evidence-sensitive on this subset if that interval sits above
zero. Otherwise the result is "not detectable at this size", not "Luna ignores
evidence". The exact McNemar test on the same cases is reported alongside.

## Reported next to it, without deciding anything

- The heuristic and rule-prior baselines on the same 50 cases (0.495 and 0.408
  when this was written, from `benchmark/`).
- Per rule: true-positive recall, false-positive specificity, and the share of
  (true-positive case, false-positive case) pairs within the rule that Luna gets
  both right.
- Abstentions and turn-limit endings on the subset.
- The primary difference recomputed with each rule left out in turn.
- Luna against the heuristic on all 80 cases, exact McNemar on per-case
  correctness. Luna's overall 0.73 against the heuristic's 0.60 hasn't had a
  paired test yet.

## Exploratory, and labelled so

The evidence-swap conditions split by whether the published detector fires on
the donor: how often the verdict held, and how often it followed the donor's
label. This is what prompted the analysis above, so it can't also confirm it.

## What this can't settle

The false-positive label means benign for T1003.001. Several false-positive
captures are simulations of other attacks that touch LSASS, and calling those
suspicious is not obviously wrong. Seven true-positive captures from two labs is
also a small base. Nothing here says anything about other techniques or other
environments.
