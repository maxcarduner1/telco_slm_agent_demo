# Public Training-Data Disclosure for OTel Hugging Face SLMs

## Scope

This report covers the Hugging Face OTel SLM models used by TelcoGPT:

- `farbodtavakkoli/OTel-Embedding-335M`
- `farbodtavakkoli/OTel-Reranker-0.6B`
- `farbodtavakkoli/OTel-LLM-1.2B-IT`

These are the domain SLMs used for the telco RAG stack: embedding, reranking, and context-grounded generation.

## Bottom Line

Yes, the training-data sources are publicly stated on Hugging Face, but only at an aggregate/source-class level. The model and dataset cards say the models were trained on telecom-focused OTel data curated by 100+ domain experts, starting from roughly 1.1M raw samples and filtered to 326,767 higher-confidence examples across the OTel dataset family.

The public documentation does not appear to disclose the exact raw source documents, document versions, individual URLs, full provenance for every sample, or the full raw corpus. Hugging Face states that the released datasets are derived QA, retrieval, reranking, and safety examples rather than the raw source documents.

## Publicly Stated Source Categories

The Hugging Face model and dataset cards list these source classes and contributors:

| Source class | Contributor | Raw samples |
| --- | --- | ---: |
| arXiv telecom papers, 3GPP standards, telecom Wikipedia, telecom Common Crawl pages | Yale University | 681,172 |
| GSMA Permanent Reference Documents, Discover portal, mixed telecom documents | GSMA | 158,006 |
| IETF RFC series | NetoAI | 100,751 |
| Industry whitepapers | Khalifa University | 62,000 |
| O-RAN specifications across working groups 1, 2, 4, 5, 6, 7, 8, 9, 10 | University of Leeds | 58,565 |
| O-RAN documents across working groups | The University of Texas at Dallas | 42,000 |
| Total raw samples |  | ~1,102,494 |

After cleaning, the public cards state the corpus was reduced to 326,767 higher-confidence examples across the OTel dataset family.

## Dataset Used By Each Model

### `OTel-Embedding-335M`

- Hugging Face model card: https://huggingface.co/farbodtavakkoli/OTel-Embedding-335M
- Training dataset: `OTel-Embedding`
- Dataset card: https://huggingface.co/datasets/farbodtavakkoli/OTel-Embedding
- Base model: `BAAI/bge-large-en-v1.5`
- Public dataset schema: `anchor`, `positive`, `negative_1` through `negative_5`
- Purpose: contrastive retrieval training for telecom document retrieval.

The `OTel-Embedding` dataset card says each sample contains an anchor query, one positive passage, and five hard negative passages. Hard negatives are mined from the corpus using dense retrieval.

### `OTel-Reranker-0.6B`

- Hugging Face model card: https://huggingface.co/farbodtavakkoli/OTel-Reranker-0.6B
- Training dataset: `OTel-Reranker`
- Dataset card: https://huggingface.co/datasets/farbodtavakkoli/OTel-Reranker
- Base model: `Qwen/Qwen3-0.6B`
- Public dataset schema: `sentence_0`, `sentence_1`, `label`
- Purpose: pointwise cross-encoder relevance scoring for query-passage pairs.

The `OTel-Reranker` dataset card says labels are continuous relevance scores derived from the data generation and filtering pipeline, not necessarily direct human labels for every pair.

### `OTel-LLM-1.2B-IT`

- Hugging Face model card: https://huggingface.co/farbodtavakkoli/OTel-LLM-1.2B-IT
- Training dataset: `OTel-LLM`
- Dataset card: https://huggingface.co/datasets/farbodtavakkoli/OTel-LLM
- Base model, per current HF model card: `LiquidAI/LFM2.5-1.2B-Instruct`
- Public dataset schema: `anchor`, `prompt`, `completion`, `prompt_type`, `abstention`, positive/negative chunk counts, token counts
- Purpose: context-grounded telecom answer generation in RAG pipelines.

The `OTel-LLM` dataset card says the dataset includes both answer examples and abstention examples, where the model is trained to refuse when retrieved context is insufficient.

## Cleaning and Processing Claims

The public dataset cards describe a four-stage cleaning pipeline:

1. Heuristic filtering to remove malformed, duplicated, or low-quality entries.
2. Reranking-based filtering to discard weakly aligned pairs.
3. Embedding-based filtering to remove semantic outliers and near-duplicates.
4. Final deduplication.

## What Is Not Publicly Stated

The Hugging Face cards do not fully disclose:

- The exact list of source documents or URLs.
- Specific document version numbers for 3GPP, GSMA, O-RAN, IETF, arXiv, Common Crawl, or whitepaper sources.
- Whether all source documents are independently redistributable in raw form.
- Per-sample provenance back to a source document.
- The full raw corpus; the released artifacts are derived training examples.
- Independent third-party validation of the cleaning pipeline or labels.

## Practical Interpretation

For customer-facing or governance discussions, the strongest accurate statement is:

> The OTel Hugging Face SLMs publicly document aggregate telco training-data sources, contributors, cleaning methodology, schemas, and derived datasets. The stated sources include 3GPP, O-RAN, GSMA, IETF RFCs, arXiv telecom papers, telecom Wikipedia/Common Crawl pages, industry whitepapers, and related telecom documents. However, the exact raw documents and per-example provenance are not publicly disclosed; the released datasets are derived examples for LLM, embedding, reranking, and safety training.

## Source Links

- OTel Hugging Face publisher page: https://huggingface.co/farbodtavakkoli
- `OTel-Embedding-335M` model card: https://huggingface.co/farbodtavakkoli/OTel-Embedding-335M
- `OTel-Reranker-0.6B` model card: https://huggingface.co/farbodtavakkoli/OTel-Reranker-0.6B
- `OTel-LLM-1.2B-IT` model card: https://huggingface.co/farbodtavakkoli/OTel-LLM-1.2B-IT
- `OTel-Embedding` dataset card: https://huggingface.co/datasets/farbodtavakkoli/OTel-Embedding
- `OTel-Reranker` dataset card: https://huggingface.co/datasets/farbodtavakkoli/OTel-Reranker
- `OTel-LLM` dataset card: https://huggingface.co/datasets/farbodtavakkoli/OTel-LLM
- `OTel-Safety` dataset card: https://huggingface.co/datasets/farbodtavakkoli/OTel-Safety
