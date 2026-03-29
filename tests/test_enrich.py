from __future__ import annotations

import pytest

from far_ai_brain.nodes.enrich import enrich_node


class TestEnrichNode:
    @pytest.mark.asyncio
    async def test_returns_empty_suggestions(self):
        state = {"extractions": []}
        result = await enrich_node(state)
        assert result["split_suggestions"] == []
        assert result["group_suggestions"] == []

    @pytest.mark.asyncio
    async def test_empty_extractions(self):
        state = {"extractions": [{"assets_to_create": []}]}
        result = await enrich_node(state)
        assert result["split_suggestions"] == []
        assert result["group_suggestions"] == []

    @pytest.mark.asyncio
    async def test_with_assets_still_returns_empty(self):
        state = {
            "extractions": [{
                "assets_to_create": [{
                    "temp_asset_id": "tmp_001",
                    "description": "Server Rack",
                    "individual_cost_with_tax": 500000,
                }],
            }],
        }
        result = await enrich_node(state)
        assert result["split_suggestions"] == []
        assert result["group_suggestions"] == []
