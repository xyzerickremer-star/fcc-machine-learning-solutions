# AuditAgent/Bachelorarbeit Transfer Note

- **Core idea:** The FCC project models each book as a sparse vector of user ratings and retrieves nearest neighbors with cosine distance. AuditAgent can use the same pattern for similarity search over audit artifacts.
- **AuditAgent mapping:** Replace `book -> user-rating vector` with `finding/control/document -> feature vector` (e.g., embeddings, categorical risk tags, control IDs, severity, process area). Nearest-neighbor retrieval can surface prior similar findings, relevant controls, or reusable evidence.
- **Bachelorarbeit angle:** This is a compact, explainable baseline for case-based reasoning in audit support: vectorize historical audit cases, filter noisy/low-signal records, index with KNN/cosine, and evaluate whether retrieved neighbors improve consistency and traceability of recommendations.
- **Caveat:** KNN quality depends on feature design and sparsity filtering; production AuditAgent use should add metadata filters, recency weighting, and validation against expert-labeled similar cases.
