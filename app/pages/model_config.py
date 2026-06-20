"""模型配置页面 - 添加/编辑/删除大模型配置"""

import streamlit as st
from src.models.model_manager import ModelConfig


st.set_page_config(page_title="模型配置", page_icon="settings", layout="wide")
st.markdown(
    """
    <style>
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main > div,
    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("模型配置")
st.caption("添加、编辑或删除大模型 API 配置。支持任何 OpenAI 兼容接口的大模型。")


mm = st.session_state.model_manager

# ── 已配置的模型列表 ──
st.header("已配置的模型")

if not mm.configs:
    st.info("还没有配置任何模型，请在下方向添加。")
else:
    for i, config in enumerate(mm.configs):
        with st.expander(f"{'启用' if config.is_active else '停用'} - {config.name} ({config.provider})", expanded=False):
            cols = st.columns([3, 1])
            with cols[0]:
                new_name = st.text_input("显示名称", value=config.name, key=f"name_{i}")
                new_provider = st.text_input("提供商", value=config.provider, key=f"prov_{i}")
                new_key = st.text_input("API 密钥", value=config.api_key or "",
                                        type="password" if config.api_key else "default", key=f"key_{i}")
                new_url = st.text_input("API 地址", value=config.base_url or "", key=f"url_{i}")
                new_model = st.text_input("模型名称", value=config.model_name or "", key=f"model_{i}")
            with cols[1]:
                if st.button("启用/停用", key=f"toggle_{i}"):
                    mm.toggle_active(config.name)
                    st.rerun()
                if st.button("保存修改", key=f"save_{i}", type="primary"):
                    updated = ModelConfig(name=new_name, provider=new_provider,
                                          api_key=new_key, base_url=new_url,
                                          model_name=new_model, is_active=config.is_active)
                    mm.add_model(updated)
                    st.success(f"已更新: {new_name}")
                    st.rerun()
                if st.button("删除", key=f"del_{i}"):
                    mm.remove_model(config.name)
                    st.rerun()

# ── 添加新模型 ──
st.divider()
st.header("添加新模型")

with st.form("add_model_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        new_name = st.text_input("显示名称 *", placeholder="例如: DeepSeek V3")
        new_provider = st.text_input("提供商", placeholder="例如: DeepSeek、Xiaomi、Kimi")
        new_url = st.text_input("API 地址 *", placeholder="https://api.deepseek.com/v1")
    with col2:
        new_model = st.text_input("模型名称 *", placeholder="deepseek-chat")
        new_key = st.text_input("API 密钥 *", type="password", placeholder="sk-...")
    submitted = st.form_submit_button("添加模型", use_container_width=True, type="primary")
    if submitted:
        if not new_name or not new_url or not new_model:
            st.error("显示名称、API 地址和模型名称为必填项。")
        else:
            config = ModelConfig(name=new_name, provider=new_provider or "自定义",
                                 api_key=new_key or "", base_url=new_url, model_name=new_model)
            mm.add_model(config)
            st.success(f"已添加模型: {new_name}")
            st.rerun()

# ── 快捷添加预设模板 ──
st.divider()
st.header("快捷添加")

presets = {
    "DeepSeek": {"provider": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model_name": "deepseek-chat"},
    "小米 Mimo": {"provider": "Xiaomi", "base_url": "https://api.minimax.chat/v1", "model_name": "mi-mimo"},
    "月之暗面 Kimi": {"provider": "Moonshot", "base_url": "https://api.moonshot.cn/v1", "model_name": "moonshot-v1-8k"},
    "OpenAI": {"provider": "OpenAI", "base_url": "https://api.openai.com/v1", "model_name": "gpt-4o-mini"},
}

preset_cols = st.columns(len(presets))
for (name, info), col in zip(presets.items(), preset_cols):
    with col:
        with st.container(border=True):
            st.subheader(name)
            st.caption(f"API 地址: {info['base_url']}")
            st.caption(f"模型名称: {info['model_name']}")
            if st.button(f"添加 {name}", key=f"preset_{name}", use_container_width=True):
                exists = any(c.name == name for c in mm.configs)
                if not exists:
                    mm.add_model(ModelConfig(name=name, provider=info["provider"],
                                             base_url=info["base_url"], model_name=info["model_name"]))
                    st.success(f"已添加 {name}")
                    st.rerun()
                else:
                    st.info(f"{name} 已存在。")
