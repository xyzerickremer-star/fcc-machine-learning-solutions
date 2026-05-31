# AuditAgent/Bachelorarbeit Transfer Note

## What transfers from this FCC regression project

This project demonstrates a compact regression pipeline that maps structured input factors to an expected numeric cost. The same pattern is directly useful for AuditAgent research prototypes where the target is not insurance cost but a continuous audit-planning quantity.

## AuditAgent mapping

- **Input features:** replace `age`, `bmi`, `smoker`, and `region` with audit-context variables such as system criticality, number of findings, control maturity, evidence volume, data-source complexity, process owner availability, and prior incident history.
- **Target variable:** replace insurance charges with a measurable audit outcome, for example expected audit effort in hours, expected remediation cost, residual risk score, or predicted testing workload.
- **Feature engineering:** keep interpretable interaction terms. Examples: high criticality × weak control maturity, many findings × low automation coverage, regulated data × external vendor dependency.
- **Model role:** use the model as a decision-support estimator for prioritization and planning, not as an autonomous audit decision-maker.
- **Evaluation:** MAE remains easy to explain in a thesis context because it is expressed in the same unit as the target, e.g. hours, euros, or score points.

## Bachelorarbeit relevance

The example can be cited as a minimal baseline for quantitative prediction in an AuditAgent architecture: collect structured audit metadata, train a reproducible regression model, evaluate with a transparent error metric, and use predictions to support risk/cost/effort prioritization while preserving human review.
