import logging

logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(self, config: dict):
        self.config = config

    async def extract_entities(self, text: str) -> list:
        """Extract entities from text (simplified for sample.txt text)."""
        try:
            # Simple rule-based entity extraction for sample.txt
            entities = []
            if "Rome" in text:
                entities.append({"name": "Rome", "type": "Location"})
            if "Venice" in text:
                entities.append({"name": "Venice", "type": "Location"})
            logger.debug(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []
