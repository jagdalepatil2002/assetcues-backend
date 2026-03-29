from __future__ import annotations

from far_ai_brain.prompts.classification import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT,
    MULTIPAGE_CLASSIFICATION_PROMPT,
)
from far_ai_brain.prompts.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    FOOTER_EXTRACTION_PROMPT,
    FULL_PAGE_EXTRACTION_PROMPT,
    HEADER_EXTRACTION_PROMPT,
    MULTI_PAGE_EXTRACTION_PROMPT,
    TABLE_EXTRACTION_PROMPT,
    TARGETED_REREAD_AMOUNT_PROMPT,
    TARGETED_REREAD_TABLE_PROMPT,
    TARGETED_REREAD_TEXT_PROMPT,
)
from far_ai_brain.prompts.group_validation import GROUP_VALIDATION_PROMPT_TEMPLATE
from far_ai_brain.prompts.split_research import (
    SPLIT_RESEARCH_PROMPT_TEMPLATE,
    SPLIT_RESEARCH_SYSTEM_PROMPT,
)
from far_ai_brain.prompts.verification import (
    VERIFICATION_PROMPT_TEMPLATE,
    VERIFICATION_SYSTEM_PROMPT,
)


class TestPromptsNonEmpty:
    def test_classification_prompts(self):
        assert len(CLASSIFICATION_SYSTEM_PROMPT) > 50
        assert len(CLASSIFICATION_USER_PROMPT) > 5
        assert len(MULTIPAGE_CLASSIFICATION_PROMPT) > 50

    def test_extraction_prompts(self):
        assert len(EXTRACTION_SYSTEM_PROMPT) > 100
        assert len(HEADER_EXTRACTION_PROMPT) > 50
        assert len(TABLE_EXTRACTION_PROMPT) > 50
        assert len(FOOTER_EXTRACTION_PROMPT) > 50
        assert len(FULL_PAGE_EXTRACTION_PROMPT) > 50
        assert len(MULTI_PAGE_EXTRACTION_PROMPT) > 50

    def test_targeted_reread_prompts(self):
        assert len(TARGETED_REREAD_AMOUNT_PROMPT) > 10
        assert len(TARGETED_REREAD_TEXT_PROMPT) > 10
        assert len(TARGETED_REREAD_TABLE_PROMPT) > 20

    def test_verification_prompts(self):
        assert len(VERIFICATION_SYSTEM_PROMPT) > 50
        assert len(VERIFICATION_PROMPT_TEMPLATE) > 20

    def test_split_research_prompts(self):
        assert len(SPLIT_RESEARCH_SYSTEM_PROMPT) > 50
        assert len(SPLIT_RESEARCH_PROMPT_TEMPLATE) > 50

    def test_group_validation_prompt(self):
        assert len(GROUP_VALIDATION_PROMPT_TEMPLATE) > 50


class TestPromptPlaceholders:
    def test_multipage_has_page_count(self):
        assert "{page_count}" in MULTIPAGE_CLASSIFICATION_PROMPT

    def test_multi_page_extraction_has_page_count(self):
        assert "{page_count}" in MULTI_PAGE_EXTRACTION_PROMPT

    def test_verification_has_fields_json(self):
        assert "{fields_json}" in VERIFICATION_PROMPT_TEMPLATE

    def test_split_research_placeholders(self):
        assert "{asset_name}" in SPLIT_RESEARCH_PROMPT_TEMPLATE
        assert "{category}" in SPLIT_RESEARCH_PROMPT_TEMPLATE
        assert "{value}" in SPLIT_RESEARCH_PROMPT_TEMPLATE
        assert "{vendor}" in SPLIT_RESEARCH_PROMPT_TEMPLATE

    def test_group_validation_placeholders(self):
        assert "{child_name}" in GROUP_VALIDATION_PROMPT_TEMPLATE
        assert "{parent_name}" in GROUP_VALIDATION_PROMPT_TEMPLATE
        assert "{child_value}" in GROUP_VALIDATION_PROMPT_TEMPLATE
        assert "{parent_value}" in GROUP_VALIDATION_PROMPT_TEMPLATE

    def test_split_research_format_works(self):
        result = SPLIT_RESEARCH_PROMPT_TEMPLATE.format(
            asset_name="Server",
            category="IT Equipment",
            value=500000,
            vendor="Dell",
        )
        assert "Server" in result
        assert "500000" in result

    def test_group_validation_format_works(self):
        result = GROUP_VALIDATION_PROMPT_TEMPLATE.format(
            child_name="Laptop Bag",
            child_value=2000,
            child_category="Accessories",
            parent_name="Laptop",
            parent_value=80000,
            parent_category="IT Equipment",
        )
        assert "Laptop Bag" in result
        assert "80000" in result

    def test_verification_format_works(self):
        result = VERIFICATION_PROMPT_TEMPLATE.format(
            fields_json='{"grand_total": 64900}',
        )
        assert "64900" in result
