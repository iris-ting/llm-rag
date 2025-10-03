# Step 4 – Parameter Comparison (Minimal Submission)

## Best Overall

- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (dim 384)
- **Selection**: `concat`, **Top-K**: 10
- **Scores**: **F1 60.92**, EM 53.00
- **Speed**: 1.687 sec/question

## Best by Strategy

- `concat` → `sentence-transformers/all-MiniLM-L6-v2` (dim 384), Top-K 10: F1 60.92, EM 53.00
- `mmr` → `sentence-transformers/all-MiniLM-L6-v2` (dim 384), Top-K 10: F1 59.95, EM 53.00

## Top Results (sorted by F1)

| Embedding                                                  |   Dim |   Top-K | Strategy   |   EM |    F1 |   Sec/Q |
|:-----------------------------------------------------------|------:|--------:|:-----------|-----:|------:|--------:|
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |      10 | concat     |   53 | 60.92 |   1.687 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |      10 | mmr        |   53 | 59.95 |   1.532 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |      10 | concat     |   49 | 56.59 |   2.177 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |       5 | concat     |   48 | 56.13 |   0.774 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |      10 | mmr        |   48 | 55.69 |   2.028 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |       5 | mmr        |   47 | 54.05 |   0.896 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |       3 | concat     |   45 | 52.85 |   0.394 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |       5 | concat     |   44 | 51.37 |   1.025 |
| sentence-transformers/all-MiniLM-L6-v2                     |   384 |       5 | mmr        |   44 | 50.72 |   0.714 |
| sentence-transformers/distiluse-base-multilingual-cased-v2 |   512 |       3 | concat     |   44 | 50.18 |   0.541 |

## Brief Analysis (≤500 words)

We compared two embedding sizes (384d vs. 512d) and two passage selection strategies (simple concatenation vs. MMR) at Top-K ∈ {3,5,10}. The best overall configuration is shown above.  On this dataset, the best 512d run scored 4.33 lower F1 than the best 384d run. Increasing Top-K generally improved recall, but very large contexts risk truncation; MMR helped reduce redundancy compared to plain concatenation. We therefore recommend the best-overall configuration for this task, while balancing speed (sec/question) against accuracy.