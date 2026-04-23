"""Tool for retrieving relevant business rules from the indexed guides."""

from framework.agent import Tool
from retrieval.retriever import RulesRetriever

_retriever = RulesRetriever()


def search_rules(query: str, k: int = 5) -> str:
    """Search the business-rules index and return matching rule chunks."""
    results = _retriever.retrieve(query, k=k)
    if not results:
        return "No relevant rules found for this query."
    parts = []
    for r in results:
        parts.append(
            f"[Score: {r.score:.3f}] {r.chunk.format_for_prompt()}"
        )
    return "\n\n---\n\n".join(parts)


SEARCH_RULES: Tool = Tool(
    name="search_rules",
    description=(
        "Search the business-rules guides for rules relevant to the question. "
        "Call this before writing a query whenever the question may involve "
        "domain-specific thresholds, exclusions, definitions, or calculations "
        "(e.g. 'on-time', 'net charge', 'completed flight', 'qualifying loan'). "
        "The query should be keywords or a short phrase describing the rule you need."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keywords or a short phrase describing the business rule to look up. "
                    "Examples: 'on-time flight definition', 'charge exclusion rules', "
                    "'net charge volume calculation'."
                ),
            },
            "k": {
                "type": "integer",
                "description": "Number of rule chunks to return (default 5, max 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    function=search_rules,
)
