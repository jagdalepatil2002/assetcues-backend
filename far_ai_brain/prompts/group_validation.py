"""
Group validation prompts — validates parent-child asset grouping suggestions.
"""

GROUP_VALIDATION_PROMPT_TEMPLATE = """In fixed asset accounting, should this asset be grouped as a child/accessory of the proposed parent?

Child Asset: "{child_name}" (₹{child_value}, {child_category})
Parent Asset: "{parent_name}" (₹{parent_value}, {parent_category})

Common valid groupings:
- Installation/setup charges → parent equipment
- Carrying case/bag → parent device
- Cable/adapter/charger → parent device
- AMC/warranty charges → parent equipment
- UPS/stabilizer → parent heavy equipment (only if purchased together)

Answer ONLY in this exact format:
YES — [reason the child belongs with the parent]
or
NO — [reason the child should be independent]"""
