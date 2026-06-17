# 学术文献分析助手

一个基于 Streamlit 的学术文献分析工作台，支持论文上传、PDF/Word/Markdown 阅读、论文搜索、多论文对比、研究空白分析、引用分析、代码执行和多模型配置。

## 1. 本地环境准备

建议使用 Python 3.10 或 3.11。

```powershell
git clone <你的仓库地址>
cd academic-lit-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

如果你使用 Conda：

```powershell
conda create -n academic-lit-agent python=3.10 -y
conda activate academic-lit-agent
pip install -r requirements.txt
```

## 2. 配置模型

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入自己的 API Key。不要把 `.env` 提交到 GitHub。

也可以启动网页后，在“模型配置”页面手动添加 OpenAI 兼容接口，例如 DeepSeek、OpenAI、Kimi 等。

## 3. 启动项目

```powershell
streamlit run app/streamlit_app.py
```

默认会打开：

```text
http://localhost:8501
```

如果端口被占用：

```powershell
streamlit run app/streamlit_app.py --server.port 8502
```

## 4. 推荐目录说明

```text
app/                  Streamlit 页面
app/pages/chat.py     文献工作台主页面
app/pages/model_config.py  模型配置页面
src/agent/            Agent 与 LLM 客户端
src/tools/            7 个业务工具
src/memory/           会话记忆
src/models/           模型配置管理
data/                 本地运行数据，不建议提交
docs/                 文档和模板
tests/                测试目录
```

## 5. 多人协作流程

第一次拿到项目：

```powershell
git clone <仓库地址>
cd academic-lit-agent
pip install -r requirements.txt
```

开发新功能时，不要直接改 `main`：

```powershell
git checkout -b feature/你的功能名
```

提交前先检查：

```powershell
python -m compileall app src tests
```

提交代码：

```powershell
git add .
git commit -m "feat: 简短描述你的功能"
git push origin feature/你的功能名
```

然后在 GitHub 上创建 Pull Request，让同学 review 后再合并。

## 6. 提交前注意

不要提交：

- `.env`
- API Key
- `data/model_configs.json`
- `data/papers.db`
- 本地上传的论文
- 浏览器测试截图
- `__pycache__`

建议每次提交前运行：

```powershell
git status
```

确认没有把敏感文件或大文件加进去。

## 7. 简单部署方式

### 方式 A：同学本地运行

这是最简单、最适合课程小组开发的方式。每个同学 clone 仓库，安装依赖，自己配置 `.env` 或在网页里配置模型。

### 方式 B：服务器运行

服务器上安装 Python 后：

```bash
git clone <仓库地址>
cd academic-lit-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

然后通过：

```text
http://服务器IP:8501
```

访问。

如果服务器长期运行，建议使用 `tmux`、`screen` 或 systemd 管理进程。

## 8. 常见问题

### 页面能打开，但模型不回答

检查：

- 模型配置是否启用
- API Key 是否正确
- Base URL 是否是 OpenAI 兼容接口
- 当前网络是否能访问模型服务

### PDF 不能预览

确认安装了 PyMuPDF：

```powershell
pip install pymupdf
```

### Semantic Scholar 搜索失败

可能是外部 API 网络或限流问题，稍后重试即可。其他本地功能不受影响。

