"""Tool 7: Safe Python code executor for data analysis and visualization."""

import threading, time as time_mod
import threading, time as time_mod
import sys, io, textwrap, traceback, json, os, tempfile, base64, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.agent.tool_registry import BaseTool


PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "plots")


class CodeExecutorTool(BaseTool):
    name = "code_executor"
    description = "Execute Python code in a restricted sandbox for data analysis, statistics, and visualization. Available libraries: json, math, statistics, collections, matplotlib, numpy (if installed). All outputs (text, plots) are captured and returned."
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute. Use print() for text output. Use plt.savefig() to save plots."},
            "timeout": {"type": "integer", "description": "Execution timeout in seconds", "default": 10},
        },
        "required": ["code"],
    }

    def run(self, code: str = "", timeout: int = 10) -> str:
        os.makedirs(PLOT_DIR, exist_ok=True)

        # ── prepare safe globals ──
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool,
            "dict": dict, "enumerate": enumerate, "float": float,
            "int": int, "isinstance": isinstance, "len": len,
            "list": list, "max": max, "min": min, "print": print,
            "range": range, "round": round, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "type": type,
            "zip": zip, "map": map, "filter": filter, "reversed": reversed,
            "set": set, "True": True, "False": False, "None": None,
        }
        safe_globals = {
            "__builtins__": safe_builtins,
            "plt": plt,
            "json": json,
            "math": math,
        }
        # Try to import optional deps
        for mod_name in ["statistics", "collections", "numpy"]:
            try:
                safe_globals[mod_name] = __import__(mod_name)
            except ImportError:
                pass

        # ── capture output ──
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sys.stderr = io.StringIO()

        output = ""
        exec_done = threading.Event()
        exec_error = []

        def run_code():
            nonlocal output
            try:
                exec_globals = safe_globals
                exec_locals = {}
                exec(textwrap.dedent(code), exec_globals, exec_locals)
                output = sys.stdout.getvalue()
            except Exception:
                output = traceback.format_exc()
            finally:
                exec_done.set()

        t = threading.Thread(target=run_code, daemon=True)
        t_start = time_mod.time()
        t.start()
        t.join(timeout=int(timeout))
        timed_out = not exec_done.is_set()

        if timed_out:
            output = f"Execution timed out after {timeout}s. Code was terminated."
        else:
            stdout_val = sys.stdout.getvalue()
            if output == "" and stdout_val:
                output = stdout_val

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # ── check for saved plots ──
        plot_refs = []
        if "plt" in safe_globals:
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for idx, fig in enumerate(figs):
                plot_path = os.path.join(PLOT_DIR, f"plot_{idx}.png")
                fig.savefig(plot_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                if os.path.exists(plot_path):
                    plot_refs.append(plot_path)

        result_parts = [f"Code execution completed in {timeout}s timeout window."]
        if output:
            result_parts.append(f"--- Output ---\n{output}")
        if plot_refs:
            result_parts.append(f"--- Plots saved ---")
            for pr in plot_refs:
                result_parts.append(f"  Plot: {pr}")

        if not output and not plot_refs:
            result_parts.append("(No output produced)")

        return "\n".join(result_parts)

