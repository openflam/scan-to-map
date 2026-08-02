# Scene-memory sensitivity experiments

This directory contains a non-destructive workflow for measuring how errors in
the constructed scene memory propagate to benchmark performance. The first
implemented perturbation is missing components.

## Missing components: experimental design

Use the original ScanNet++ scene memories as the clean reference. For every
scene and seed, assign each component a deterministic SHA-256 rank. A deletion
condition removes a prefix of that ranking, so the 5%, 10%, 20%, and 30% masks
are nested within a seed. This makes comparisons across severity paired and
prevents unrelated random masks from obscuring the trend.

The default design is:

- one clean baseline;
- deletion rates of 5%, 10%, 20%, and 30%;
- two independent masks per non-zero rate (seeds 0 and 1);
- all ScanNet++ questions in `benchmark/data`;
- the full tool configuration, `search_dist_around_image_exec`;
- the same reasoning model for every condition.

Deletion is uniform over the components in each scene, not over benchmark
answers. Each copied question is annotated as `all_retained`,
`partially_removed`, `all_removed`, or `no_referenced_components`.
Report both:

1. Overall performance versus the fraction of the scene memory removed. This is
   the population-level robustness result.
2. Performance stratified by reference status. The retained-target stratum
   measures collateral error propagation through context and spatial reasoning;
   the removed-target stratum measures graceful behavior when necessary evidence
   is absent.

The preparation script deletes an object consistently from
`component_captions.json`, `bbox_corners.json`, `crops/manifest.json`,
crop-image access, and `connected_components.json` when present. It does not
renumber retained components. Thus expected and predicted component IDs remain
comparable across conditions.

## Safety and artifact layout

Existing scene memories and benchmark artifacts are treated as immutable.

- Original data remains in `outputs/<dataset>` and `benchmark/data`.
- Perturbed memories use new top-level aliases such as
  `outputs/missing_components_scannetpp_..._bp0500_s000`.
- Questions, results, metrics, tables, and plots are written below the new
  `benchmark/missing_components` tree.
- Preparation fails if the experiment directory or any intended dataset alias
  already exists.
- Answer generation can fail on any pre-existing result, and the runner uses
  that mode by default.
- Summarization refuses to overwrite an existing analysis directory.
- Neither `benchmark/plots` nor `benchmark/tables` is used by this workflow.

## End-to-end command

The primary entry point runs preparation, PostGIS loading, answer generation,
metric computation, aggregation, and plotting in order. Its defaults use two
seeds and one tool configuration, producing nine conditions and 621 top-level
GPT-5.4 benchmark queries for the current 69-question benchmark.

Rebuild and start the Docker services so the search-server image contains the
metric and plotting dependencies:

    docker compose up --build --detach

Validate the complete plan without creating artifacts or making model calls:

    docker compose exec search-server python /app/benchmark-eval/sensitivity_analysis/missing_components/experiment.py --dry-run

Run the complete experiment with the same defaults:

    docker compose exec search-server python /app/benchmark-eval/sensitivity_analysis/missing_components/experiment.py

The final analysis directory contains per-question data, aggregate CSV tables,
a JSON summary, and PDF plots for absolute performance and change from the clean
baseline. Use `--disable-ai-judge` for deterministic metrics only. Use
`--resume --reuse-existing-db-tables` to continue an interrupted, matching run;
neither flag permits overwriting the final analysis directory.

## Individual stages

The lower-level scripts remain available for inspecting or running individual
stages.

First inspect the plan without creating data:

```bash
python benchmark-eval/sensitivity_analysis/missing_components/prepare.py --dry-run
```

Prepare the scene-memory aliases and condition question files:

```bash
python benchmark-eval/sensitivity_analysis/missing_components/prepare.py
```

The runner defaults to printing all commands without executing them:

```bash
python benchmark-eval/sensitivity_analysis/missing_components/run.py
```

With PostGIS and the search server available, load only the new alias tables,
then generate answers and metrics:

```bash
python benchmark-eval/sensitivity_analysis/missing_components/run.py --stage load-db
python benchmark-eval/sensitivity_analysis/missing_components/run.py --stage answers
python benchmark-eval/sensitivity_analysis/missing_components/run.py --stage metrics
```

Alternatively, run those stages in order:

```bash
python benchmark-eval/sensitivity_analysis/missing_components/run.py --stage all
```

A partial run can be continued explicitly with `--resume`. Existing PostGIS
alias tables are never silently replaced; use `--reuse-existing-db-tables`
only after verifying they came from the current manifest.

Finally, aggregate the completed metric files:

```bash
python benchmark-eval/sensitivity_analysis/missing_components/summarize.py
```

The summarizer emits per-question data, overall summaries, paired changes from
the clean baseline, reference-status summaries, and PDF degradation curves. Its
95% intervals use a scene-seed cluster bootstrap, keeping questions from the
same perturbed scene together.

For a cheaper deterministic smoke study, omit the LLM judge while computing
metrics:

```bash
python benchmark-eval/sensitivity_analysis/missing_components/run.py \
  --stage metrics \
  --disable-ai-judge
python benchmark-eval/sensitivity_analysis/missing_components/summarize.py \
  --metrics "F1 Score" Recall
```

Do not mix deterministic-only and AI-judge metric files in one experiment
directory.

## Recommended reporting

For each metric and deletion rate, report the mean, 95% cluster-bootstrap
interval, and change from the clean baseline. The most compact main table is:

| Removed | AI-Judge | Component F1 | Component recall |
|---:|---:|---:|---:|
| 0% | score | score | score |
| 5% | score (change) | score (change) | score (change) |
| 10% | score (change) | score (change) | score (change) |
| 20% | score (change) | score (change) | score (change) |
| 30% | score (change) | score (change) | score (change) |

Add a small conditional table comparing `all_retained` and
`all_removed`. Include the number of observations in every stratum; at low
deletion rates the removed-target subset will naturally be smaller.

Interpret the two views separately. A shallow decline for retained targets
supports robustness to incomplete distractor/context memory. A sharp decline
when target evidence is removed is expected and should be described as an
information-limit result, alongside whether the system avoids hallucinating
the missing entity.

## Follow-up perturbations for the full reviewer comment

Keep the same rates, seeds, questions, model, tool configuration, artifact
layout, and aggregation. Changing only the perturbation allows direct comparison
of error types.

### Incorrect merges

At each rate, greedily select disjoint component pairs from a deterministic
ranking. Use two protocols:

- spatial merges: pair each selected component with its nearest neighbor;
- semantic-confuser merges: pair components with similar labels/captions when
  available.

Keep one canonical ID, concatenate the two captions with neutral punctuation,
union the boxes, combine crop-manifest entries, and remove the absorbed ID.
Record a source-to-canonical ID map so ground-truth references can be scored
both strictly and after merge-aware remapping. Report both scores; their gap
separates retrieval failure from unavoidable identity loss.

### Caption noise

Use at least two controlled corruptions:

- deletion noise: remove the object head noun or a fixed fraction of caption
  tokens;
- substitution noise: replace the head noun with a different scene label,
  sampled deterministically.

Apply noise to a uniform component subset at the same rates. Keep geometry and
images unchanged. Stratify questions by whether a referenced component's
caption was corrupted. Substitution is more diagnostic than generic character
noise because it represents a plausible confident captioning error.

### Bounding-box perturbations

Perturb box centers and sizes while preserving valid eight-corner boxes:

- center translation with Gaussian magnitude normalized by the box diagonal;
- multiplicative size noise in log space, with positive side lengths.

Use severities such as 2.5%, 5%, 10%, and 20% of the scene or box scale, with
five seeds. Keep captions, identities, and images fixed. Report spatial
questions separately from entity-search questions and include a geometry metric
such as mean 3D IoU between clean and perturbed boxes.

### Cross-error summary

For the paper, use one plot with normalized performance change versus corruption
severity and one table at a representative severity. Avoid combining all errors
in the primary sensitivity study: one-factor-at-a-time perturbations make the
source of degradation identifiable. A small optional mixed-error condition can
then show whether effects compound approximately additively.
