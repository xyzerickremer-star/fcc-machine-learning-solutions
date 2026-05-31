# Transfer Note: Rock Paper Scissors → AuditAgent / Bachelorarbeit

## freeCodeCamp concept

The project implements an adaptive Rock/Paper/Scissors agent. The agent observes prior opponent moves, estimates the opponent's next move using small predictive models, and selects the counter-move that maximizes expected wins.

Core ideas:

- Sequential decision-making
- Pattern recognition from historical behavior
- Simple opponent modeling / hypothesis scoring
- Feedback loop: observe → predict → act → update

## AuditAgent analogy

The same high-level loop can be mapped to audit assistance:

1. **Observe**: read historical booking patterns, document submissions, PBC responses, and prior audit outcomes.
2. **Predict**: estimate the next relevant risk, likely document class, or expected follow-up action.
3. **Act**: suggest a sample item, PBC request, classification, or next audit procedure.
4. **Update**: incorporate reviewer feedback and actual findings to improve prioritization.

This is not yet a production-grade ML model; it is a compact demonstration of data-driven agent behavior and adaptive decision support.

## Bachelorarbeit relevance

Useful framing:

> A KI-supported audit assistant can be conceptualized as a decision-support loop that observes engagement data, recognizes behavioral or transactional patterns, and recommends the next audit action while keeping the auditor in control.

Potential connection to stichprobenorientierte Einzelfallprüfung:

- Use historical transaction patterns to flag unusual items.
- Prioritize single-case testing based on risk signals instead of pure random selection.
- Compare predicted vs. actual auditor feedback to improve rule/model quality.

## Product note

For AuditAgent, this maps best to a first lightweight module:

- input: population of bookings + metadata + prior labels
- output: ranked list of suspicious or representative items
- reviewer action: approve/reject sample proposal
- feedback: reviewer decision becomes training/evaluation signal
