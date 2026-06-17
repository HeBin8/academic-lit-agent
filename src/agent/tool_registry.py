"""Generic tool registration framework.

Each tool is a class with:
  - name: unique tool identifier
  - description: LLM-facing description (for ReAct thought prompting)
  - parameters: JSON Schema describing expected arguments
  - run(**kwargs): execute the tool, return str result
"""

from typing import Any, Optional
import json


class ToolSpec:
    """Metadata for a tool, used in the system prompt to describe it to the LLM."""

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema dict

    def to_prompt_block(self) -> str:
        """Format as a readable block for the system prompt."""
        params_desc = []
        for prop_name, prop in self.parameters.get("properties", {}).items():
            req = "required" if prop_name in self.parameters.get("required", []) else "optional"
            params_desc.append(f"  - {prop_name} ({req}): {prop.get('description', '')}")
        lines = [
            f"## {self.name}",
            f"Description: {self.description}",
            "Parameters:",
        ]
        lines.extend(params_desc)
        return "\n".join(lines)


class BaseTool:
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)

    def run(self, **kwargs) -> str:
        """Execute the tool. Return a textual result for the LLM to observe."""
        raise NotImplementedError


class ToolRegistry:
    """Registry of available tools that the agent can call."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def spec_list(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def tool_descriptions(self) -> str:
        """Full block of tool descriptions for the system prompt."""
        blocks = [s.to_prompt_block() for s in self.spec_list()]
        return "\n\n".join(blocks)

    def parse_action(self, text: str) -> Optional[dict]:
        """Parse Action: ToolName | param=value | ... from LLM output."""
        import re
        match = re.search(r"Action:\s*(\w+)", text)
        if not match:
            return None
        name = match.group(1)
        if name not in self._tools:
            return None
        tool = self._tools[name]
        params = tool.parameters.get("properties", {}).keys()
        kwargs = {}
        for param in params:
            pm = re.search(rf"{param}\s*[:=]\s*\"(.+?)\"", text)
            if pm:
                kwargs[param] = pm.group(1)
            else:
                pm = re.search(rf"{param}\s*[:=]\s*(.+)", text)
                if pm:
                    kwargs[param] = pm.group(1).strip()
        return {"tool": name, "kwargs": kwargs}

    def execute(self, action: dict) -> str:
        """Execute a parsed action dict."""
        tool = self._tools.get(action["tool"])
        if not tool:
            return f"Error: unknown tool '{action['tool']}'"
        try:
            return tool.run(**action["kwargs"])
        except Exception as e:
            return f"Error executing {action['tool']}: {e}"
