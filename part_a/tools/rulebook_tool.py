from __future__ import annotations

from pathlib import Path

from docx import Document


TOPIC_KEYWORDS = {
    "temporal correlation": ["temporal correlation", "correlation window", "hourly matching", "monthly averaging"],
    "hourly matching": ["hourly matching", "rule 3", "hour"],
    "monthly averaging": ["monthly averaging", "rule 2", "month"],
    "curtailment and outages": ["curtailment", "outages", "availability = 0", "no data"],
    "grid purchases": ["grid purchases", "power purchase agreement", "hourly attribution"],
    "reporting granularity": ["reporting granularity", "hourly granularity", "monthly granularity"],
    "rfnbo": ["rfnbo", "compliance"],
}


class RulebookTool:

    def __init__(self, rulebook_path: str | Path | None = None):

        if rulebook_path is None:
            rulebook_path = Path(__file__).resolve().parents[2] / "data_pack" / "compliance_rulebook.docx"

        self.source = Path(rulebook_path)
        document = Document(self.source)
        self.lines = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]

    def _match_topics(self, query: str) -> list[str]:
        lowered = query.lower().strip()
        topics = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                topics.append(topic)
        return topics

    def lookup(self, query: str) -> dict:

        lowered = query.lower().strip()
        topics = self._match_topics(lowered)

        matches: list[str] = []
        if topics:
            for topic in topics:
                for line in self.lines:
                    if any(keyword in line.lower() for keyword in TOPIC_KEYWORDS[topic]):
                        matches.append(line)
        else:
            for line in self.lines:
                if lowered in line.lower():
                    matches.append(line)

        unique_matches = list(dict.fromkeys(matches))

        if not unique_matches:
            return {
                "ok": False,
                "query": query,
                "topic": None,
                "matches": [],
                "source": str(self.source),
                "message": "No relevant rule found in the supplied rulebook.",
            }

        return {
            "ok": True,
            "query": query,
            "topic": topics[0] if topics else None,
            "matches": unique_matches,
            "source": str(self.source),
            "message": "Relevant rule text retrieved from the supplied rulebook.",
        }
