# KoBioMed-Llama

> 🚧 **Under active development** — early-stage scaffold. Model weights and training pipelines are being prepared.

KoBioMed-Llama is a Korean biomedical large language model project with PubMed/PDF retrieval-augmented generation (RAG), built to support life-science research at the MSCSL Lab (Chungnam National University, Korea) and Mitos Therapeutics.

## Goals

- Korean-capable biomedical LLM for life-science Q&A, literature summarization, and mechanism analysis
- RAG pipeline over PubMed abstracts and lab PDF collections
- Evaluation on biomedical benchmarks

## Repository layout

- `src/kobiomed_llama/` — model training / fine-tuning code
- `src/rag/` — retrieval pipeline (PubMed, PDF ingestion, vector store)
- `configs/` — training and RAG configuration files
- `evals/` — evaluation scripts and datasets
- `data/` — data preparation notes (raw data is not committed)

## Status & roadmap

- [x] Project scaffold
- [ ] Data curation (PubMed, domain PDFs)
- [ ] Continued pre-training / instruction tuning
- [ ] RAG pipeline
- [ ] Benchmark evaluation
- [ ] Model weights release

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).

---

🇰🇷 **한국어 안내**: 본 프로젝트는 생명과학 연구 지원을 위한 한국어 바이오메디컬 LLM 프로젝트입니다. PubMed 초록과 연구실 PDF 문헌에 대한 검색증강생성(RAG)을 포함하며, 현재 초기 개발 단계입니다.
