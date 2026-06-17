import pathlib, sys

path = pathlib.Path("D:/CodexProjects/academic-lit-agent/app/pages/chat.py")
content = path.read_text(encoding="utf-8")

# Find and replace CSS section
marker = 'st.markdown("""'
idx = content.find(marker)
if idx < 0:
    print("ERROR: CSS marker not found")
    sys.exit(1)

# Find the full st.markdown block
end_idx = content.find('st.markdown("""', idx + 10)
if end_idx < 0:
    end_idx = content.find("with st.form(", idx + 10)

# Find the closing of the markdown call 
close_idx = content.find(")", end_idx - 100)
if close_idx < 0:
    close_idx = content.find("unsafe_allow_html=True)", idx)

new_css = """    st.markdown(\"\"\"
    <style>
    div[data-testid="stForm"] {
        border: 1px solid #e0e0e0 !important;
        border-radius: 24px !important;
        padding: 2px 12px !important;
        background: #fff !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stForm"]:focus-within {
        border-color: #1677ff !important;
    }
    div[data-testid="stForm"] div[data-testid="stTextInput"] input {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        font-size: 14px;
        padding: 8px 4px !important;
    }
    div[data-testid="stForm"] button[kind="secondary"] {
        background: none !important;
        border: none !important;
        font-size: 18px !important;
        padding: 0 6px !important;
        min-width: 32px !important;
        min-height: 0 !important;
        height: 36px !important;
        color: #555 !important;
        box-shadow: none !important;
        border-radius: 50% !important;
    }
    div[data-testid="stForm"] button[kind="secondary"]:hover {
        background: #f5f5f5 !important;
        color: #1677ff !important;
    }
    div[data-testid="stForm"] button[kind="primary"] {
        border-radius: 50% !important;
        width: 36px !important;
        min-width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        font-size: 18px !important;
    }
    div[data-testid="stForm"] [data-testid="column"] { padding: 0 2px !important; display: flex !important; align-items: center !important; }
    div[data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
    div[data-testid="stPopoverBody"] { min-width: 220px; }
    </style>
    \"\"\", unsafe_allow_html=True)"""

# Find the exact block boundaries more carefully
start_marker = '<style>'
start_idx = content.find(start_marker, idx)
end_style = content.find('</style>', start_idx)
close_def = content.find('unsafe_allow_html=True)', end_style)

if start_idx >= 0 and close_def >= 0:
    before = content[:start_idx - 8]  # 8 = len('    '), the indentation before st.markdown
    after = content[close_def + len('unsafe_allow_html=True)'):]
    content = before + new_css + after
    path.write_text(content, encoding="utf-8")
    print(f"CSS replaced successfully. New file size: {len(content)}")
else:
    print(f"Could not find CSS boundaries: start={start_idx}, close_def={close_def}")
