import sqlite3
import logging 
from datetime import datetime, timezone
from pathlib import Path
 
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "usage.db"

class PricingCalculator:
    """
    Owns ONE responsibility: knowing model prices and computing cost.
    Nothing about databases, logging, or how usage gets stored — just
    "given a model and token counts, what does this cost?"
    """

    def __init__(self):
        # Per-million-token rates, in USD. Source: OpenAI pricing page,
        # verified against current rates as of mid-2026 — ALWAYS confirm
        # against https://openai.com/api/pricing/ before relying on this
        # for real budgeting, since OpenAI revises these periodically.
        self.model_pricing = {
            "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
            "gpt-4o": {"input_per_million": 2.50, "output_per_million": 10.0}
        }

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Returns 0.0 for unrecognized models rather than raising —
        an unknown model shouldn't crash the caller's request.
        """
        pricing = self.model_pricing.get(model)
        if pricing is None:
            logger.warning(f"No pricing entry for model '{model}', treating cost as 0.0")
            return 0.0

        input_cost = (prompt_tokens / 1000000) * pricing["input_per_million"]
        output_cost = (completion_tokens / 1000000) * pricing["output_per_million"]
        return round(input_cost + output_cost, 8)


class UsageRepository:
    """
    Owns ONE responsibility: persisting and retrieving usage records.
    Nothing about pricing or business logic — just storage. Swapping
    SQLite for Postgres later means changing ONLY this class.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                cached INTEGER NOT NULL
            )""")
        conn.commit()
        conn.close()

    def insert(self, endpoint: str, model: str, prompt_tokens: int, completion_tokens: int, cost: float, cached: bool)->None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO usage_log
                (timestamp, endpoint, model, prompt_tokens, completion_tokens, estimated_cost_usd, cached)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                endpoint,
                model,
                prompt_tokens,
                completion_tokens,
                cost,
                int(cached),
            ),
        )
        conn.commit()
        conn.close()

    
    def total_cost_since(self, since_iso_timestamp: str) -> float:
        """
        Returns summed estimated_cost_usd for all rows after the given
        ISO timestamp. Useful later for budgeting (e.g. "today's spend").
        """
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM usage_log WHERE timestamp >= ?",
            (since_iso_timestamp,),
        ).fetchone()
        conn.close()
        return row[0]
    

class CostLogger:
    """
    The only class other code should actually call. Orchestrates
    PricingCalculator + UsageRepository — callers just say "log this
    call," without knowing or caring how pricing or storage work
    internally.
    """
 
    def __init__(self, pricing_calculator: PricingCalculator, repository: UsageRepository):
        self._pricing = pricing_calculator
        self._repository = repository
 
    def log_usage(
        self,
        endpoint: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool = False,
    ) -> None:
        """
        Cached responses are logged with $0.0 cost (no OpenAI call was
        actually made), but token counts are still recorded for
        visibility into "would-be" usage. Never raises — a logging
        failure should never break the actual API response.
        """
        cost = 0.0 if cached else self._pricing.estimate_cost(model, prompt_tokens, completion_tokens)
 
        try:
            self._repository.insert(
                endpoint=endpoint,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                cached=cached,
            )
        except sqlite3.Error as e:
            logger.warning(f"Failed to log usage: {e}")


# Single shared instance, reused across service modules
cost_logger = CostLogger(
    pricing_calculator=PricingCalculator(),
    repository=UsageRepository(),
)