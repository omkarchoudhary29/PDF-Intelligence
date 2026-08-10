"""Entity Normalization and Deduplication Service for Enterprise Knowledge Navigator.

Provides deterministic canonical normalization, punctuation and case normalization,
entity type validation, and alias resolution.
"""

import re
from typing import Optional, Set

# Standard entity categories supported by the knowledge graph
VALID_ENTITY_TYPES: Set[str] = {
    "Machine",
    "Equipment",
    "Component",
    "System",
    "Process",
    "Material",
    "Product",
    "Project",
    "Organization",
    "Department",
    "Policy",
    "Regulation",
    "Person",
    "Location",
    "Technical concept",
    "Standard",
    "Safety hazard",
    "Maintenance procedure",
    "Measurement",
    "Other",
}

# Type aliases / normalization mapping
ENTITY_TYPE_ALIASES = {
    "device": "Equipment",
    "tool": "Equipment",
    "hardware": "Equipment",
    "component": "Component",
    "part": "Component",
    "subsystem": "System",
    "system": "System",
    "engine": "Machine",
    "motor": "Machine",
    "turbine": "Machine",
    "material": "Material",
    "company": "Organization",
    "institution": "Organization",
    "corp": "Organization",
    "vendor": "Organization",
    "guideline": "Policy",
    "compliance": "Regulation",
    "procedure": "Process",
    "workflow": "Process",
    "method": "Process",
    "concept": "Technical concept",
    "technology": "Technical concept",
    "spec": "Standard",
    "specification": "Standard",
    "iso": "Standard",
    "hazard": "Safety hazard",
    "safety": "Safety hazard",
    "maintenance": "Maintenance procedure",
    "inspection": "Maintenance procedure",
    "measurement": "Measurement",
    "metric": "Measurement",
}


class EntityNormalizationService:
    """Provides deterministic string normalization and category validation for extracted entities."""

    @staticmethod
    def normalize_name(raw_name: str) -> str:
        """Normalize an entity name for indexing and deduplication.

        Rules:
        - Case fold to lower
        - Strip whitespace and collapse multiple spaces to a single space
        - Normalize dashes/hyphens to a standard ASCII hyphen
        - Strip surrounding quotes, brackets, and terminal punctuation
        - Preserve alphanumeric technical identifiers (e.g. 'AX-450', 'ISO-9001')
        """
        if not raw_name:
            return ""

        text = raw_name.strip()
        # Remove surrounding quotes, brackets, parentheses
        text = re.sub(r"^[\"'`\(\[\{]+|[\"'`\)\]\}]+$", "", text).strip()
        # Normalize non-standard hyphens (em-dash, en-dash) to standard minus
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)
        # Lowercase for normalized lookup key
        text = text.lower()
        # Strip trailing punctuation (. , : ;)
        text = text.rstrip(".,;:-_")
        return text.strip()

    normalize_entity_name = normalize_name

    @staticmethod
    def clean_canonical_name(raw_name: str) -> str:
        """Clean human-facing canonical display name while preserving proper title casing."""
        if not raw_name:
            return ""

        text = raw_name.strip()
        text = re.sub(r"^[\"'`\(\[\{]+|[\"'`\)\]\}]+$", "", text).strip()
        text = re.sub(r"\s+", " ", text)
        text = text.rstrip(".,;:-_")
        return text.strip()

    @staticmethod
    def validate_and_normalize_type(raw_type: str) -> str:
        """Validate and map entity type to standard knowledge graph category."""
        if not raw_type:
            return "Other"

        cleaned = raw_type.strip()
        # Direct match check (case-insensitive)
        for valid in VALID_ENTITY_TYPES:
            if cleaned.lower() == valid.lower():
                return valid

        # Alias lookup
        mapped = ENTITY_TYPE_ALIASES.get(cleaned.lower())
        if mapped and mapped in VALID_ENTITY_TYPES:
            return mapped

        return "Other"

    @classmethod
    def are_aliases(cls, name_a: str, name_b: str) -> bool:
        """Check if two names are deterministic aliases (e.g. 'Unit 4' vs 'Unit-4')."""
        norm_a = cls.normalize_name(name_a)
        norm_b = cls.normalize_name(name_b)

        if norm_a == norm_b:
            return True

        # Check hyphen vs space equivalence
        if norm_a.replace("-", " ") == norm_b.replace("-", " "):
            return True
        if norm_a.replace("-", "") == norm_b.replace("-", ""):
            return True

        return False


entity_normalization_service = EntityNormalizationService()
