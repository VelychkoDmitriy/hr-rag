# Retrieval: 15 questions

| configuration | recall@1 | recall@5 | recall@10 | MRR@10 | outdated chunk at top-1 | ms / query | missed in top-5 |
|---|---|---|---|---|---|---|---|
| fixed chunks, vector | 0.73 | 0.87 | 1.00 | 0.785 | 0 | 24 | r05, r15 |
| vector | 0.67 | 0.93 | 1.00 | 0.794 | 2 | 25 | r05 |
| BM25 | 0.53 | 0.73 | 0.80 | 0.628 | 1 | 6 | r01, r03, r05, r13 |
| hybrid (RRF) | 0.73 | 0.87 | 1.00 | 0.814 | 2 | 33 | r05, r13 |
| hybrid + rerank | 0.80 | 0.93 | 1.00 | 0.878 | 2 | 33 | r05 |
| hybrid + rerank, current versions only | 0.93 | 1.00 | 1.00 | 0.950 | 0 | 78 | — |
| + query rewrite (uk + en), current only | 0.93 | 1.00 | 1.00 | 0.950 | 0 | 349 | — |

Ranks of the first correct chunk per question:

| question | fixed chunks, vector | vector | BM25 | hybrid (RRF) | hybrid + rerank | hybrid + rerank, current versions only | + query rewrite (uk + en), current only |
|---|---|---|---|---|---|---|---|
| r01 (paraphrase) | 1 | 1 | 6 | 1 | 1 | 1 | 1 |
| r02 (paraphrase) | 1 | 2 | 4 | 1 | 1 | 1 | 1 |
| r03 (paraphrase) | 1 | 1 | — | 1 | 1 | 1 | 1 |
| r04 (exact) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| r05 (paraphrase) | 10 | 6 | — | 9 | 6 | 4 | 4 |
| r06 (exact) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| r07 (cross_lingual) | 3 | 2 | 2 | 2 | 2 | 1 | 1 |
| r08 (outdated) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| r09 (table) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| r10 (table) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| r11 (similar) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| r12 (exact) | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| r13 (multi_condition) | 5 | 4 | — | 10 | 1 | 1 | 1 |
| r14 (similar) | 1 | 1 | 2 | 1 | 1 | 1 | 1 |
| r15 (exact) | 7 | 2 | 1 | 2 | 2 | 1 | 1 |
