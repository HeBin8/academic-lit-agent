"""Academic Literature Analysis Agent - ReAct + Reflection reasoning loop."""

import re, json, time
from typing import Optional
from src.agent.llm_client import LLMClient
from src.agent.tool_registry import ToolRegistry
from src.memory.conversation_memory import ConversationMemory


SYSTEM_PROMPT_TEMPLATE = """You are an academic literature analysis assistant. Your goal is to help users research academic topics by searching, reading, and analyzing papers.

You operate in a ReAct (Reasoning + Acting) loop:
1. Think: analyze the user''s request and plan the next step
2. Action: call one tool with parameters
3. Observe: read the tool''s output
4. Repeat until you have enough information
5. Answer: provide a comprehensive final response

Available tools:
{tool_descriptions}

Format your response exactly like this:

Thought: <your reasoning about what to do next>
Action: tool_name | param1=value1 | param2=value2

Parameter rules:
- Use | to separate parameters
- Use = or : to assign values
- String values can be quoted or unquoted
- Example: Action: search_papers | query=retrieval augmented generation | limit=10

When you have enough information to answer, use:

Thought: I have all the information needed.
Final Answer: <your comprehensive response to the user>

Guidelines:
- Always show your reasoning in Thought before each Action.
- Use multiple tool calls when the question requires it.
- For literature comparison, first search for relevant papers, then use literature_comparator.
- For research gap analysis, collect paper data first, then use research_gap_analyzer.
- After a tool returns an error, try once more or use a different tool.
- If you cannot find enough information, say so honestly in your Final Answer.
- Cite papers by their paper ID and title when referencing them.
"""


def _build_action_prompt() -> str:
    return (
        "\n\nRemember to respond with:\n"
        "Thought: <reasoning>\n"
        "Action: tool_name | param1=value1 | param2=value2\n\n"
        "Or if done:\n"
        "Thought: I have all the information needed.\n"
        "Final Answer: <response>"
    )


class AcademicLitAgent:
    """Main agent that orchestrates the ReAct loop with multi-turn memory."""

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry,
                 max_steps: int = 15, reflection: bool = True):
        self.llm = llm_client
        self.tools = tool_registry
        self.memory = ConversationMemory()
        self.max_steps = max_steps
        self.reflection = reflection

        # Build system prompt with tool descriptions
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=self.tools.tool_descriptions()
        )

    # ── public interface ─────────────────────────────────────────

    def process_message(self, user_message: str, document_context: str = "") -> dict:
        """Process a single user message through the ReAct loop.

        Returns: {
            "response": str,        # Final answer
            "trace": list[dict],    # Step-by-step reasoning trace
            "tokens_used": int,
            "steps": int,
            "tools_called": list[str],
        }
        """
        self.memory.new_session(self.system_prompt)
        if document_context:
            self.memory.messages.insert(1, {"role": "system", "content": document_context})
        self.memory.add_user(user_message)

        trace = []
        tools_called = []
        tokens_used = 0

        for step in range(1, self.max_steps + 1):
            # Get LLM response
            history = self.memory.get_history()
            reply = self.llm.chat(history)
            tokens_used += self.llm.count_tokens(reply)

            # Check for Final Answer
            if "Final Answer:" in reply or "FINAL ANSWER:" in reply.upper():
                # Extract the final answer text
                fa_match = re.search(
                    r"(?:Final Answer|FINAL ANSWER)\s*:\s*(.+)",
                    reply, re.DOTALL
                )
                final_answer = fa_match.group(1).strip() if fa_match else reply
                trace.append({
                    "step": step,
                    "type": "final_answer",
                    "content": final_answer,
                })
                self.memory.add_assistant(reply)
                break

            # Parse Action
            action_match = re.search(
                r"Action:\s*(\w+)\s*(?:\||)(.*?)(?=\n|$)",
                reply, re.DOTALL
            )

            if not action_match:
                # No action found - try to see if this is a thinking-only response
                trace.append({
                    "step": step,
                    "type": "thought",
                    "content": reply,
                })
                # Ask the LLM to provide an action
                self.memory.add_assistant(reply)
                self.memory.add_assistant(
                    "I need to take an action to make progress. "
                    "Please specify Action: tool_name | param=value"
                )
                continue

            tool_name = action_match.group(1)
            param_text = action_match.group(2).strip()

            # Parse parameters from the action line
            kwargs = self._parse_params(param_text)

            # Log thought
            thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", reply, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""

            trace.append({
                "step": step,
                "type": "action",
                "thought": thought,
                "tool": tool_name,
                "params": kwargs,
            })

            # Execute tool
            action = {"tool": tool_name, "kwargs": kwargs}
            result = self.tools.execute(action)
            tools_called.append(tool_name)

            trace.append({
                "step": step,
                "type": "observation",
                "tool": tool_name,
                "result": result[:2000],
            })

            # Add to memory
            self.memory.add_assistant(reply)
            self.memory.add_tool_result(tool_name, result)

            # Self-reflection step
            if self.reflection and step > 1:
                reflection_prompt = (
                    "Reflection: Based on the observation above, "
                    "is my approach working? If not, I should adjust."
                )
                self.memory.add_assistant(reflection_prompt)
        else:
            # Max steps reached without Final Answer
            trace.append({
                "step": self.max_steps + 1,
                "type": "final_answer",
                "content": (
                    "I have completed my analysis with the available steps. "
                    "The main findings are summarized above."
                ),
            })

        # Get the last assistant message as our final answer
        final_answer = "Analysis complete. See trace for details."
        for m in reversed(self.memory.messages):
            if m["role"] == "assistant":
                fa_match = re.search(
                    r"(?:Final Answer|FINAL ANSWER)\s*:\s*(.+)",
                    m["content"], re.DOTALL
                )
                if fa_match:
                    final_answer = fa_match.group(1).strip()
                else:
                    # Use the last substantial assistant message
                    content = m["content"]
                    # Strip the ReAct format to get the answer part
                    if "Thought:" in content and "Action:" not in content:
                        thought_m = re.search(r"Thought:(.+?)(?=Final|$)", content, re.DOTALL)
                        if thought_m:
                            final_answer = thought_m.group(1).strip()
                    elif not content.startswith("Thought:"):
                        final_answer = content[:1000]
                break

        return {
            "response": final_answer,
            "trace": trace,
            "推理过程": trace,
            "tokens_used": tokens_used,
            "steps": min(step + 1, self.max_steps),
            "tools_called": list(dict.fromkeys(tools_called)),  # unique, preserves order
        }

    # ── internal helpers ─────────────────────────────────────────

    def _parse_params(self, param_text: str) -> dict:
        """Parse 'key=value | key2=value2' format into dict."""
        kwargs = {}
        if not param_text:
            return kwargs
        pairs = re.split(r"\s*\|\s*", param_text)
        for pair in pairs:
            pair = pair.strip()
            if not pair:
                continue
            m = re.match(r"(\w+)\s*[=:]\s*(.+)", pair)
            if m:
                key = m.group(1)
                value = m.group(2).strip().strip("\"'")
                kwargs[key] = value
        return kwargs

    def reset_conversation(self):
        """Start a fresh conversation session."""
        self.memory = ConversationMemory()

