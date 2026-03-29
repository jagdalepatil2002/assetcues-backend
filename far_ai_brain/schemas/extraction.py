"""
Full extraction output schema — the JSON structure returned by the VLM.

Every field extracted from the invoice uses ConfidenceField[T] so the
frontend always knows how confident the AI was about each value.
"""
from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ConfidenceField(BaseModel, Generic[T]):
    """A field value paired with its confidence score and source provenance."""
    value: Optional[T] = None
    confidence: Optional[float] = None
    source_region: Optional[str] = None  # "header", "table", "footer", "full", "reread"
    bbox: Optional[dict] = None


# ── Section models ──


class ExtractionMeta(BaseModel):
    extraction_id: str = ""
    model_used: str = ""
    model_version: Optional[str] = None
    timestamp: Optional[str] = None
    processing_time_ms: Optional[int] = None
    overall_confidence: Optional[float] = None
    image_quality_score: Optional[float] = None
    document_language_detected: Optional[str] = None
    pages_processed: int = 1
    source_image_refs: list[str] = Field(default_factory=list)


class VendorDetails(BaseModel):
    vendor_name: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    vendor_gstin: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    vendor_pan: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    vendor_address: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    vendor_phone: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    vendor_email: ConfidenceField[str] = Field(default_factory=ConfidenceField)


class BuyerDetails(BaseModel):
    buyer_name: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    buyer_gstin: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    buyer_address: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    ship_to_address: ConfidenceField[str] = Field(default_factory=ConfidenceField)


class InvoiceHeader(BaseModel):
    invoice_number: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    invoice_date: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    due_date: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    po_number: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    po_date: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    grn_number: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    delivery_note_number: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    challan_number: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    payment_terms: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    currency: ConfidenceField[str] = Field(default_factory=ConfidenceField)


class LineItem(BaseModel):
    line_index: int = 0
    description: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    hsn_sac_code: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    quantity: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    unit: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    unit_price: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    discount_percent: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    taxable_amount: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    cgst_rate: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    cgst_amount: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    sgst_rate: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    sgst_amount: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    igst_rate: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    igst_amount: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    line_total: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    serial_numbers_listed: list[str] = Field(default_factory=list)


class Totals(BaseModel):
    subtotal_before_tax: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    total_cgst: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    total_sgst: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    total_igst: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    total_tax: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    rounding_off: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    grand_total: ConfidenceField[float] = Field(default_factory=ConfidenceField)
    amount_in_words: ConfidenceField[str] = Field(default_factory=ConfidenceField)


class BankDetails(BaseModel):
    bank_name: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    account_number: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    ifsc_code: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    branch: ConfidenceField[str] = Field(default_factory=ConfidenceField)


class WarrantyInfo(BaseModel):
    warranty_period: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    warranty_start_date: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    amc_period: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    amc_start_date: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    support_contact: ConfidenceField[str] = Field(default_factory=ConfidenceField)


class AssetToCreate(BaseModel):
    temp_asset_id: str = ""
    source_line_index: int = 0
    asset_name: str = ""
    quantity_index: int = 1
    quantity_total: int = 1
    individual_cost_before_tax: Optional[float] = None
    individual_tax: Optional[float] = None
    individual_cost_with_tax: Optional[float] = None
    serial_number: Optional[str] = None
    suggested_category: Optional[str] = None
    suggested_sub_category: Optional[str] = None
    suggested_asset_class: Optional[str] = None
    suggested_model: Optional[str] = None
    suggested_make: Optional[str] = None
    date_of_acquisition: Optional[str] = None
    audit_indicator: Optional[str] = None  # "physical" | "non_physical"
    audit_method: Optional[str] = None
    group_action: str = "none"  # "none" | "suggest_group_with_parent"
    group_parent_temp_id: Optional[str] = None
    group_reason: Optional[str] = None
    split_action: str = "none"  # "none" | "suggest_split"
    split_components: Optional[list[dict]] = None
    confidence_overall: Optional[float] = None
    # Bulk/unit-aware fields
    unit_of_measure: Optional[str] = None  # "Nos", "Kg", "L", etc.
    bulk_quantity: Optional[float] = None  # Original qty for bulk items
    is_bulk_asset: bool = False  # True if measured in weight/volume


class TextBlock(BaseModel):
    text: str = ""
    bbox: Optional[dict] = None
    is_handwritten: bool = False
    page_index: int = 0


class StampSignature(BaseModel):
    description: str = ""
    bbox: Optional[dict] = None
    type: str = ""  # "stamp" | "signature" | "watermark"
    page_index: int = 0


class RawExtraction(BaseModel):
    all_text_blocks: list[TextBlock] = Field(default_factory=list)
    stamps_and_signatures: list[StampSignature] = Field(default_factory=list)
    tables_raw: list[dict] = Field(default_factory=list)
    other_visible_info: list[str] = Field(default_factory=list)
    handwritten_notes: list[TextBlock] = Field(default_factory=list)
    full_text_dump: str = ""


class MathCheck(BaseModel):
    line_totals_valid: bool = True
    subtotal_valid: bool = True
    tax_calculations_valid: bool = True
    grand_total_valid: bool = True
    issues: list[str] = Field(default_factory=list)


class FormatCheck(BaseModel):
    gstin_valid: Optional[bool] = None
    pan_valid: Optional[bool] = None
    dates_valid: bool = True
    hsn_valid: Optional[bool] = None
    issues: list[str] = Field(default_factory=list)


class ValidationResults(BaseModel):
    math_check: MathCheck = Field(default_factory=MathCheck)
    format_checks: FormatCheck = Field(default_factory=FormatCheck)
    cross_model_verification: Optional[dict] = None


class AIDecision(BaseModel):
    decision: str = ""
    reason: str = ""
    confidence: Optional[float] = None


# ── Top-level extraction output ──


class ExtractionOutput(BaseModel):
    """The complete extraction output for a single invoice."""
    extraction_meta: ExtractionMeta = Field(default_factory=ExtractionMeta)
    document_type: ConfidenceField[str] = Field(default_factory=ConfidenceField)
    vendor_details: VendorDetails = Field(default_factory=VendorDetails)
    buyer_details: BuyerDetails = Field(default_factory=BuyerDetails)
    invoice_header: InvoiceHeader = Field(default_factory=InvoiceHeader)
    line_items: list[LineItem] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    bank_details: BankDetails = Field(default_factory=BankDetails)
    warranty_amc_info: WarrantyInfo = Field(default_factory=WarrantyInfo)
    assets_to_create: list[AssetToCreate] = Field(default_factory=list)
    raw_complete_extraction: RawExtraction = Field(default_factory=RawExtraction)
    validation_results: ValidationResults = Field(default_factory=ValidationResults)
    ai_reasoning_log: list[AIDecision] = Field(default_factory=list)
