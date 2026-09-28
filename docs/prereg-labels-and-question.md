# Checking the labels and the question, written down before the run

The within-rule result (`docs/prereg-within-rule.md`) says Luna's verdict follows
the capture. What it can't say is how many of Luna's remaining misses are misses,
for two reasons that live in this repo rather than in the model:

- "False positive" means not T1003.001 credential dumping. None of the 17
  false-positive captures is mapped to T1003 by Security-Datasets, but some of
  them open or patch LSASS on purpose (over-pass-the-hash writes a key into
  LSASS memory, for one).
- The prompt asks whether "the activity the rule describes really happened".
  For an LSASS-access rule on such a capture, it did.

Three checks, fixed here before any of them has a result. One costs money
(about $0.32, run with `--budget-usd 1`); the others read existing files.

## A. The label review

`benchmark/label-review.yml` lists the 17 false-positive captures with the
dataset's own title, description, ATT&CK mapping and tool, and two questions per
capture: is it credential theft from LSASS memory, and does it touch LSASS.
It is filled in by hand from that sheet and the dataset's public pages, and
committed before any per-capture model result is looked at. A handful of
per-capture results have already come up while writing earlier docs (the
heuristic's misses on `empire_over_pth_patch_lsass`, for instance); the review
is still done from the sheet, not from those.

A false-positive capture is **clear** if the answer to credential theft is
"no", and **ambiguous** if it's "yes" or "unsure". The labels themselves are not
changed; the split is only used to report results both ways.

## B. The technique question

One Luna run, condition `technique-question`: the reference run with a single
phrase changed, so a true positive is "credential theft from LSASS memory,
ATT&CK T1003.001" instead of "the activity the rule describes". All 80 cases,
three repeats, same settings as Luna's reference run.

**Primary:** macro-F1 of `technique-question` minus `reference` on all 80
cases, paired, majority over repeats, 95% interval from 2,000 capture
resamples (stratified by label, seed 0), exactly as `score.analysis` reports it
for any condition against the reference. The wording is said to matter if the
interval excludes zero, in either direction.

**Reported with it:**

- false-positive specificity under both wordings, on clear and on ambiguous
  captures separately;
- true-positive recall under both wordings, so a model that simply says
  "false positive" more often doesn't look like an improvement;
- every existing Luna condition re-scored with the ambiguous captures left out,
  next to the original scores.

This is a check of what the benchmark measures, not a better prompt. Whatever
the result, the reference run stays the headline number; the new wording was
chosen after seeing results and is never reported as an improvement to Luna.

## C. What the verdicts cite

From the `technique-question` run only (earlier runs didn't log citations): for
each accepted decisive verdict, whether it cites a process-access event
(Sysmon EventID 10, or Security 4656/4663) or only other events. Split by the
verdict and by the case's label. Descriptive, no test.

## What can't come out of this

Nothing here adds captures: 7 attack captures from two labs, one technique.
And Luna remains a model picked after the results.
