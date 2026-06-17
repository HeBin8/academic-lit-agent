---
name: latex-hutb-report
description: Generate structured LaTeX course reports following the HUTB (Hunan University of Technology and Business) template. Use when the user needs to produce a Chinese academic course report with cover page, abstract, TOC, numbered chapters, GB/T 7714-2015 bibliography, GB/T 1526 diagrams, and appendix. Template supports XeLaTeX with Chinese fonts (SimSun, SimHei).
---

# LaTeX HUTB Course Report

Full LaTeX template for HUTB course reports in `assets/`. Compile with XeLaTeX + Biber.

## Template Structure

```
assets/
  main.tex              -- Entry point, orchestrates all sub-files
  style/
    hutb_thesis.cls      -- Document class, cover info key-value system
    hutb_setup.tex       -- Packages, fonts, geometry, title formatting
  text/
    01_封面.tex           -- Cover page (logo, title, student info via \coverinfo)
    02_摘要.tex           -- Abstract + keywords
    03_目录.tex           -- Table of contents, list of tables, list of figures
    Chapter1.tex          -- Introduction / requirement analysis
    Chapter2.tex          -- Design
    Chapter3.tex          -- Implementation
    Chapter4.tex          -- Testing / evaluation / conclusion
    附录.tex              -- Appendix (code listings, additional diagrams)
  bib/
    bibtext.bib           -- Bibliography (GB/T 7714-2015 via biblatex)
  figures/
    logo.png, hutb_icon.png, hutb_title.jpg  -- School branding images
```

## Workflow for Report Generation

### Step 1: Copy Template

Copy the entire `assets/` tree into the target output directory:

```bash
cp -r assets/ /path/to/report-output/
```

### Step 2: Fill Cover Info

Edit `text/01_封面.tex` and update the `\coverinfo{}` block:

```latex
\coverinfo{
    title = 面向垂直场景的轻量级AI智能体开发与实践,
    class = 计算机技术xxxx班,
    stuname = 张三,
    stuid = 2024xxxxxx,
    teachers = 李四,
    year = 2026,
    month = 6,
    day = 16
}
```

Also update the course name on the preceding `\textbf{《课程名称》 报告}` line.

### Step 3: Write Abstract

Edit `text/02_摘要.tex`: fill the Chinese title, student name, abstract text, and 3-5 keywords.

### Step 4: Populate Chapters

- **Chapter1.tex**: Requirement analysis, application scenario, GB/T 1526 workflow diagram
- **Chapter2.tex**: System architecture, module interaction diagram, tool interface definitions
- **Chapter3.tex**: Implementation details, core code structure, key algorithm descriptions
- **Chapter4.tex**: Test cases table, evaluation metrics, optimization log, conclusion

### Step 5: Add Bibliography

Edit `bib/bibtext.bib` with GB/T 7714-2015 formatted entries. Use `export_bibtex.py` from `semantic-scholar-tools` to auto-generate entries from paper IDs.

### Step 6: Build

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Output: `main.pdf`

## GB/T 1526 Diagrams

See [references/gbt1526_diagrams.md](references/gbt1526_diagrams.md) for TikZ-based GB/T 1526 flowchart patterns suitable for agent workflows, tool-call routing, and module interaction diagrams.

## Key Formatting Notes

- Chapter titles: `\section{}` auto-formats as ` "第X章  Title"`  (SimHei, 小二)
- Subsection: `\subsection{}` (SimHei, 四号)
- Tables use `booktabs` + `tabular` with `[H]` float placement
- Citations: `\cite{}` with biblatex + gb7714-2015 style
- Chinese font: SimSun (宋体) for body; SimHei (黑体) for headings
- English font: Times New Roman
- Geometry: A4, textwidth=138mm, margins=31.8mm left/right
- Do NOT modify `style/hutb_thesis.cls` or `style/hutb_setup.tex` unless fixing a font issue
