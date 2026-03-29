"""
Document classification prompts — identify document type and plan extraction.
Runs before extraction to prime the pipeline.
"""

CLASSIFICATION_SYSTEM_PROMPT = """You are a document classification expert. Given an invoice or bill image, classify it and create an extraction plan.

Return a JSON object with these fields:
- document_type: one of "printed_invoice", "handwritten_bill", "thermal_receipt", "credit_note", "debit_note", "proforma_invoice", "delivery_challan", "quotation", "other"
- is_handwritten: true if the document contains significant handwriting
- language: primary language ("english", "hindi", "mixed", or other)
- complexity: "simple" (1 page, clear layout, standard table) or "complex" (multi-page, mixed layout, handwritten, poor quality)
- has_table: true if a structured table of line items is visible
- estimated_line_items: approximate number of line items visible
- notable_features: list of notable features like "watermark", "stamp", "multiple_signatures", "handwritten_corrections", "poor_quality", "rotated"
"""

CLASSIFICATION_USER_PROMPT = "Classify this document and create an extraction plan."

MULTIPAGE_CLASSIFICATION_PROMPT = """You are looking at thumbnails of {page_count} pages from a single PDF document.

Determine:
1. Are these pages from ONE invoice (line items continue across pages, same vendor/invoice number), or MULTIPLE SEPARATE invoices (different vendors or invoice numbers per page/section)?
2. Group the pages by invoice.

Return JSON:
{{
  "pattern": "single_invoice" or "multiple_invoices",
  "page_groups": [[0, 1, 2]] or [[0], [1, 2], [3]],
  "reasoning": "brief explanation"
}}

Page indices are 0-based."""
