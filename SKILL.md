---
name: ocr-document-understanding
description: Specialized skill for analyzing, developing, optimizing, and testing Vietnamese administrative document AI OCR and Document Understanding systems. Use when modifying OCR pipelines, PDF extraction, local AI models, semantic extraction, spelling correction, or OCR performance.
---

# OCR DOCUMENT UNDERSTANDING ENGINEERING SKILL

## 1. Mission

Act as a specialized engineer responsible for the complete development lifecycle of the Vietnamese administrative document understanding system.

The system extracts exactly five fields:
- Issuing agency.
- Document number and symbol.
- Document date.
- Document subject.
- Document type.

## 2. Repository analysis

Before making changes:
1. Inspect the repository structure.
2. Identify programming languages, frameworks and dependencies.
3. Find application entry points.
4. Map the current document processing pipeline.
5. Identify OCR engines and language models.
6. Identify inference endpoints and model loading mechanisms.
7. Identify all hardcoded business recognition logic.
8. Find existing tests and performance benchmarks.

Never assume the repository architecture without inspecting it.

## 3. Processing architecture

Evaluate the document before choosing a processing path.

PATH A – Searchable PDF:
- Extract text directly from PDF.
- Evaluate text extraction quality.
- Avoid image conversion and OCR where text extraction is valid.
- Prepare the minimum relevant content for semantic extraction.

PATH B – Scanned PDF or image:
- Use a dedicated lightweight OCR model.
- Preserve page and text-region evidence.
- Prioritize processing relevant content.
- Avoid unnecessary OCR passes.
- Escalate to additional pages or higher-quality OCR only when required.

PATH C – Uncertain extraction:
- Analyze missing or conflicting fields.
- Process only the necessary additional content.
- Do not invent information.
- Mark uncertain results for human review.

## 4. AI semantic extraction

The AI must independently understand context and identify the five fields.

Never replace semantic understanding with hardcoded business rules.

Distinguish issuing agency from recipient, signing person, advisory department and mentioned organizations.

Preserve document identifiers and dates accurately.

Prefer original subject text over generated summaries.

Classify document types using semantic understanding rather than fixed keyword matching.

## 5. Performance engineering

For every performance task:
1. Establish a baseline.
2. Measure each pipeline stage separately.
3. Identify the actual bottleneck.
4. Propose the smallest effective change.
5. Implement the change.
6. Repeat the same benchmark.
7. Compare accuracy and latency.

Prioritize:
- Model warm-up and persistent loading.
- PDF text extraction.
- OCR model selection.
- CPU inference optimization.
- Quantization.
- Prompt reduction.
- Number of model calls.
- Concurrent request management.
- Memory usage.

Do not introduce GPU-dependent architecture when the current deployment is CPU-only.

## 6. Accuracy and correction

Evaluate each extracted field independently.

Use OCR evidence and semantic context for spelling correction.

Do not silently modify agency names, identifiers, dates or numerical values without evidence.

Return null when information cannot be reliably identified.

Keep original OCR text and corrected text distinguishable.

## 7. Implementation discipline

- Reuse the existing project architecture where appropriate.
- Keep modules independently testable.
- Keep configuration outside business logic.
- Avoid unnecessary dependencies.
- Preserve existing functionality.
- Use local inference for document processing.
- Do not transmit real document content to external services.

## 8. Testing and acceptance

Test:
- Searchable PDF.
- Scanned PDF.
- Image documents.
- Multi-page documents.
- Skewed and low-quality scans.
- Different administrative layouts.
- Vietnamese diacritics and OCR errors.
- Missing and ambiguous fields.

Measure:
- Total processing latency.
- First-result latency.
- OCR duration.
- LLM duration.
- CPU and RAM.
- Accuracy of each of the five fields.
- P50 and P95 latency.

Never fabricate benchmark results.

## 9. Completion report

Every completed task must report:
1. What was analyzed.
2. What was changed.
3. Files modified.
4. Technical reasoning.
5. Tests actually executed.
6. Benchmark results when applicable.
7. Remaining issues.
8. Recommended next actions.

The goal is a production-ready, fast, accurate and maintainable on-premise document understanding system.
