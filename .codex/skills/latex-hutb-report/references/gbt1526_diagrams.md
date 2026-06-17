# GB/T 1526 Diagram Guidance for LaTeX

GB/T 1526-1989 is the Chinese national standard for information processing flowchart symbols, equivalent to ISO 5807.

## TikZ Approach (Recommended)

Use `tikz` with `shapes.geometric` and `arrows.meta` for programmatic diagrams that stay crisp in PDF.

```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

\tikzstyle{process} = [rectangle, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=blue!10]
\tikzstyle{decision} = [diamond, aspect=2, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=yellow!10]
\tikzstyle{data} = [trapezium, trapezium left angle=70, trapezium right angle=110, minimum width=3cm, minimum height=1cm, text centered, draw=black, fill=green!10]
\tikzstyle{arrow} = [thick,->,>=Stealth]
```

## GB/T 1526 Symbol Mapping

| Symbol | GB/T 1526 Name | TikZ Shape | Use |
|--------|---------------|------------|-----|
| Rectangle | Process | `rectangle` | Computation, action step |
| Diamond | Decision | `diamond` | Branch/conditional |
| Parallelogram | Data I/O | `trapezium` | Input/output |
| Ellipse/rounded rect | Terminator | `rounded rectangle` | Start/End |
| Cylinder | Database | `cylinder` | SQLite/DB access |
| Document | Document | Custom `node` | File read/write |
| Circle | Connector | `circle` | Flow junction |

## Agent Workflow Diagram Example

For the agent's ReAct loop, draw this pattern:

```
User Input -> [Start] -> [Plan/Thought] -> [Tool Call] -> [Observation] -> [Response?] -> Output
                                            ^                                  |
                                            |_____________ No _______________|
```

## In Report

Place diagrams under `figures/` as standalone `.tex` files and `\input` them:

```latex
\begin{figure}[H]
    \centering
    \input{figures/agent_workflow}
    \caption{Agent ReAct Workflow (GB/T 1526)}
    \label{fig:agent-workflow}
\end{figure}
```

For raster exports (screenshots), use `\includegraphics` instead.
