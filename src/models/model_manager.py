"""Model configuration manager - users add their own API configs."""

import json, os, pathlib
from dataclasses import dataclass, field
from typing import Optional


CONFIG_FILE = pathlib.Path(__file__).parent.parent.parent / "data" / "model_configs.json"


@dataclass
class ModelConfig:
    """A single model configuration."""
    name: str                     # Display name: "DeepSeek V3"
    provider: str                 # Provider label: "DeepSeek", "Xiaomi", "Custom"
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""          # API model identifier: "deepseek-chat"
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            name=d.get("name", ""),
            provider=d.get("provider", ""),
            api_key=d.get("api_key", ""),
            base_url=d.get("base_url", ""),
            model_name=d.get("model_name", ""),
            is_active=d.get("is_active", True),
        )


class ModelManager:
    """Manages multiple model configurations. Persisted to JSON."""

    def __init__(self):
        self.configs: list[ModelConfig] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────

    def _load(self):
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                self.configs = [ModelConfig.from_dict(c) for c in data]
            except (json.JSONDecodeError, KeyError):
                self.configs = []
        if not self.configs:
            self._add_defaults()
            self._save()

    def _add_defaults(self):
        """Pre-populate with common models as templates."""
        defaults = [
            ModelConfig(
                name="DeepSeek V3",
                provider="DeepSeek",
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com/v1",
                model_name="deepseek-chat",
            ),
            ModelConfig(
                name="Xiaomi Mimo",
                provider="Xiaomi",
                api_key=os.getenv("MIMO_API_KEY", ""),
                base_url="https://api.minimax.chat/v1",  # placeholder
                model_name="mi-mimo",
            ),
        ]
        self.configs = defaults

    def _save(self):
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = [c.to_dict() for c in self.configs]
        CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── CRUD ─────────────────────────────────────────────────────

    def list_models(self) -> list[ModelConfig]:
        return [c for c in self.configs if c.is_active]

    def get_model(self, name: str) -> Optional[ModelConfig]:
        for c in self.configs:
            if c.name == name and c.is_active:
                return c
        return None

    def add_model(self, config: ModelConfig) -> None:
        """Add or update a model config by name."""
        for i, c in enumerate(self.configs):
            if c.name == config.name:
                self.configs[i] = config
                self._save()
                return
        self.configs.append(config)
        self._save()

    def remove_model(self, name: str) -> bool:
        self.configs = [c for c in self.configs if c.name != name]
        self._save()
        return True

    def toggle_active(self, name: str) -> bool:
        for c in self.configs:
            if c.name == name:
                c.is_active = not c.is_active
                self._save()
                return c.is_active
        return False

    @property
    def active_names(self) -> list[str]:
        return [c.name for c in self.configs if c.is_active]
