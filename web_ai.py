#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 AI Web 版 v1.5
- 只跑 GGUF（禁用 transformers，避免 Python 3.14 崩溃）
- 大按钮：扫描模型 / 加载模型 / 卸载 / 上传图片 / 清空对话
- 图片上传 + 缩略图预览
- n_ctx=4096，支持较长输入
"""

import os, re, io, sys, json, time, base64, traceback, webbrowser, threading
from pathlib import Path
from threading import Thread

# ---------- 可选依赖 ----------
HAS_LLAMA = False
HAS_PIL = False

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    pass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    pass

try:
    from flask import Flask, request, Response, jsonify
except ImportError:
    print("缺少 flask：pip install flask")
    sys.exit(1)

HAS_TRANSFORMERS = False


# ==========================================================
#  <think> 流式解析器
# ==========================================================
def stream_think_filter(it):
    OT = ["<think>", "<thinking>"]
    CT = ["</think>", "</thinking>"]
    buf = ""
    mode = "text"

    def ps(s, tags):
        for t in tags:
            for i in range(1, len(t)):
                if s.endswith(t[:i]):
                    return i
        return 0

    for piece in it:
        buf += piece
        while buf:
            if mode == "text":
                e, et = -1, None
                for t in OT:
                    i = buf.find(t)
                    if i != -1 and (e == -1 or i < e):
                        e, et = i, t
                if e == -1:
                    k = ps(buf, OT)
                    n = len(buf) - k
                    if n > 0:
                        yield ("text", buf[:n])
                        buf = buf[n:]
                    break
                else:
                    if e > 0:
                        yield ("text", buf[:e])
                    buf = buf[e + len(et):]
                    mode = "think"
            else:
                e, et = -1, None
                for t in CT:
                    i = buf.find(t)
                    if i != -1 and (e == -1 or i < e):
                        e, et = i, t
                if e == -1:
                    k = ps(buf, CT)
                    n = len(buf) - k
                    if n > 0:
                        yield ("think", buf[:n])
                        buf = buf[n:]
                    break
                else:
                    if e > 0:
                        yield ("think", buf[:e])
                    buf = buf[e + len(et):]
                    mode = "text"
    if buf:
        yield (mode, buf)


# ==========================================================
#  图片缩略图
# ==========================================================
def make_thumb(data, size=200):
    if not HAS_PIL:
        return base64.b64encode(data).decode()
    try:
        im = Image.open(io.BytesIO(data))
        im.thumbnail((size, size))
        b = io.BytesIO()
        im.convert("RGB").save(b, format="JPEG", quality=75)
        return base64.b64encode(b.getvalue()).decode()
    except Exception:
        return base64.b64encode(data).decode()


# ==========================================================
#  LocalLLM
# ==========================================================
class LocalLLM:
    def __init__(self):
        self.backend = None
        self.model = None
        self.model_path = None
        self.current_name = None

    def unload(self):
        self.model = None
        self.backend = None
        self.model_path = None
        self.current_name = None

    def load_gguf(self, path, n_ctx=4096, n_threads=None):
        if not HAS_LLAMA:
            return False, "未安装 llama-cpp-python"
        if os.path.isdir(path):
            gs = sorted(f for f in os.listdir(path) if f.lower().endswith(".gguf"))
            if not gs:
                return False, "目录内无 .gguf"
            path = os.path.join(path, gs[0])
        if not os.path.isfile(path):
            return False, "文件不存在"
        try:
            n_threads = n_threads or max(1, (os.cpu_count() or 4) // 2)
            self.unload()
            self.model = Llama(model_path=path, n_ctx=n_ctx,
                               n_threads=n_threads, verbose=False)
            self.backend = "gguf"
            self.model_path = path
            return True, f"GGUF 已加载：{os.path.basename(path)}（n_ctx={n_ctx}）"
        except Exception as e:
            traceback.print_exc()
            return False, f"加载失败：{e}"

    def chat_stream(self, history, sysp=None, mx=512, temp=0.7):
        sp = sysp or "你是一个本地 AI 助手，可以聊天、写代码。回答简洁准确。"
        msgs = [{"role": "system", "content": sp}]
        for r, c in history:
            msgs.append({"role": r, "content": c})
        if self.backend == "gguf":
            st = self.model.create_chat_completion(
                messages=msgs, max_tokens=mx, temperature=temp,
                top_p=0.9, stream=True,
            )
            for ck in st:
                d = ck["choices"][0].get("delta", {})
                p = d.get("content", "")
                if p:
                    yield p


# ==========================================================
#  全局状态
# ==========================================================
llm = LocalLLM()
conversation_history = []
MAX_TURNS = 20
discovered_models = []
_mlock = threading.Lock()

uploaded_images = []
_ulock = threading.Lock()


def _add_model(n, p, t):
    with _mlock:
        for m in discovered_models:
            if m["path"] == p:
                return False
        discovered_models.append({"name": n, "path": p, "type": t})
        return True


def _list_models():
    with _mlock:
        return [dict(m) for m in discovered_models]


def _clear_models():
    with _mlock:
        discovered_models.clear()


def _add_image(it):
    with _ulock:
        uploaded_images.append(it)


def _list_images():
    with _ulock:
        return [dict(u) for u in uploaded_images]


def _clear_images():
    with _ulock:
        uploaded_images.clear()


def _rm_image(fid):
    with _ulock:
        for i, u in enumerate(uploaded_images):
            if u["id"] == fid:
                uploaded_images.pop(i)
                return True
    return False


# ==========================================================
#  递归扫描
# ==========================================================
SKIP = {"Windows", "$Recycle.Bin", "System Volume Information", "Recovery",
        "node_modules", ".git", ".svn", ".hg", "__pycache__",
        ".venv", "venv", "env", "AppData", "ProgramData", "PerfLogs"}


def scan_models(root, max_depth=8, timeout=60):
    found = []
    start = time.time()
    root = os.path.abspath(root)
    for dp, dns, fns in os.walk(root, onerror=lambda e: None):
        if time.time() - start > timeout:
            break
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dns[:] = []
            continue
        dns[:] = [d for d in dns if d not in SKIP
                  and not d.startswith(".") and not d.startswith("$")]
        is_gguf = any(f.lower().endswith(".gguf") for f in fns)
        if is_gguf:
            found.append({"name": os.path.basename(dp) or dp,
                          "path": dp, "type": "gguf"})
            dns[:] = []
    return found


# ==========================================================
#  命令
# ==========================================================
HELP_TEXT = """可用命令：
  /help /status /models /scan <路径> /load <路径> /unload /clear
  /images         列出已上传图片
  /clearimages    清空所有图片

顶部按钮：扫描模型 / 加载模型 / 卸载 / 上传图片 / 清空对话
"""


def parse_cmd(t):
    t = t.strip()
    if not t:
        return None
    if t.startswith("/"):
        ps = t[1:].split(None, 1)
        return (ps[0].lower(), ps[1].strip().strip('"') if len(ps) > 1 else "")
    if re.match(r'^[A-Za-z]:[\\/]', t) or t.startswith("\\\\"):
        return ("load", t.strip().strip('"'))
    return None


def exec_cmd(c, a):
    if c == "help":
        return HELP_TEXT
    if c == "status":
        L = [f"后端: {llm.backend or '未加载'}"]
        if llm.backend:
            L.append(f"名称: {llm.current_name}")
            L.append(f"路径: {llm.model_path}")
        L.append(f"轮数: {len(conversation_history)//2}")
        L.append(f"模型数: {len(_list_models())}")
        L.append(f"图片数: {len(_list_images())}")
        L.append(f"GGUF: {'✓' if HAS_LLAMA else '✗'}")
        L.append(f"PIL: {'✓' if HAS_PIL else '✗'}")
        return "\n".join(L)
    if c == "models":
        it = _list_models()
        if not it:
            return "（空）"
        L = [f"共 {len(it)} 个："]
        for i, m in enumerate(it, 1):
            mk = " ← 当前" if m["path"] == llm.model_path else ""
            L.append(f"{i}. [{m['type']}] {m['name']}{mk}")
        return "\n".join(L)
    if c == "images":
        it = _list_images()
        if not it:
            return "（无图片）"
        return "\n".join(f"{i}. {u['name']} ({u['size']} 字节)"
                         for i, u in enumerate(it, 1))
    if c == "clearimages":
        _clear_images()
        return "✓ 图片已清空"
    if c == "scan":
        if not a:
            return "用法：/scan <路径>"
        if not os.path.isdir(a):
            return f"✗ 不是目录"
        try:
            f = scan_models(a)
        except Exception as e:
            return f"✗ {e}"
        n = sum(1 for m in f if _add_model(m["name"], m["path"], m["type"]))
        return f"✓ 找到 {len(f)} 个，新增 {n} 个"
    if c == "load":
        if not a:
            return "用法：/load <路径>"
        t = None
        if a.lower().endswith(".gguf"):
            t = "gguf"
        elif os.path.isdir(a):
            if any(f.lower().endswith(".gguf") for f in os.listdir(a)):
                t = "gguf"
        if t is None:
            return "✗ 无法判断模型类型"
        ok, msg = llm.load_gguf(a)
        if ok:
            llm.current_name = os.path.basename(a.rstrip("/\\"))
            _add_model(llm.current_name, a, t)
        return ("✓ " if ok else "✗ ") + msg
    if c == "unload":
        llm.unload()
        return "✓ 已卸载"
    if c == "clear":
        conversation_history.clear()
        return "✓ 已清空"
    return f"未知：/{c}"


def try_math(t):
    import ast, operator
    e = t.strip().rstrip("?？=").strip()
    if not re.fullmatch(r"[0-9+\-*/%().\s^]+", e):
        return None
    e = e.replace("^", "**")
    O = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
         ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
         ast.Mod: operator.mod, ast.Pow: operator.pow,
         ast.USub: operator.neg, ast.UAdd: operator.pos}

    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in O:
            return O[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in O:
            return O[type(n.op)](ev(n.operand))
        raise ValueError()
    try:
        r = ev(ast.parse(e, mode="eval"))
        if isinstance(r, float) and r.is_integer():
            r = int(r)
        return f"{t.strip()} = {r}"
    except Exception:
        return None


# ==========================================================
#  Flask
# ==========================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>离线 AI v1.5</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
       background: #0f1116; color: #e6e8ef; height: 100vh; display: flex; overflow: hidden; }
.sidebar { width: 300px; background: #171a21; border-right: 1px solid #262b36;
           padding: 16px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
.sidebar h1 { font-size: 17px; font-weight: 600; color: #7dd3fc; }
.sidebar h2 { font-size: 11px; font-weight: 600; color: #6b7280;
              text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.status-box { background: #0f1116; border: 1px solid #262b36;
              border-radius: 8px; padding: 10px 12px; font-size: 12px;
              line-height: 1.6; color: #9ca3af; word-break: break-all; }
.status-box .ok { color: #4ade80; } .status-box .err { color: #f87171; }
.scan-input-row { display: flex; gap: 6px; margin-bottom: 6px; }
.scan-input-row input { flex: 1; background: #0f1116; border: 1px solid #262b36;
                        border-radius: 6px; color: #e6e8ef; padding: 7px 10px;
                        font-size: 12px; font-family: monospace; outline: none; }
.scan-input-row input:focus { border-color: #2563eb; }
.scan-input-row button { background: #2563eb; color: white; border: none;
                         border-radius: 6px; padding: 7px 12px; font-size: 12px;
                         font-weight: 500; cursor: pointer; white-space: nowrap; }
.model-list { display: flex; flex-direction: column; gap: 4px;
              max-height: 180px; overflow-y: auto; padding-right: 4px; }
.model-list::-webkit-scrollbar { width: 6px; }
.model-list::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
.model-item { background: #0f1116; border: 1px solid #262b36;
              border-radius: 6px; padding: 8px 10px; font-size: 12px;
              cursor: pointer; transition: all 0.15s; display: flex;
              justify-content: space-between; align-items: center; gap: 8px; }
.model-item:hover { border-color: #7dd3fc; background: #1a1f2a; }
.model-item.active { border-color: #4ade80; background: #16271a; }
.model-item .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.model-item .type { font-size: 10px; padding: 1px 6px; border-radius: 3px;
                    background: #262b36; color: #9ca3af; flex-shrink: 0; }
.model-item.active .type { background: #16a34a; color: #fff; }
.btn { background: #2563eb; border: none; color: white; padding: 7px 12px;
       border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 500; }
.btn:hover { background: #1d4ed8; }
.btn.danger { background: #dc2626; } .btn.danger:hover { background: #b91c1c; }
.btn.ghost { background: #1a1f2a; color: #9ca3af; border: 1px solid #262b36; }
.btn.ghost:hover { background: #262b36; color: #e6e8ef; }
.btn.full { width: 100%; } .btn.small { padding: 5px 9px; font-size: 11px; }
.params label { display: flex; justify-content: space-between; font-size: 12px;
                color: #9ca3af; margin-bottom: 4px; margin-top: 6px; }
.params input[type=range] { width: 100%; accent-color: #2563eb; }
.params .value { color: #7dd3fc; font-family: monospace; }
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

.action-bar { background: #171a21; border-bottom: 1px solid #262b36;
              padding: 10px 16px; display: flex; gap: 8px;
              flex-wrap: wrap; align-items: center; }
.action-bar .spacer { flex: 1; }
.abtn { display: inline-flex; align-items: center; gap: 6px;
        padding: 9px 16px; border-radius: 8px; border: 1px solid #262b36;
        background: #1a1f2a; color: #e6e8ef; font-size: 13px;
        font-weight: 600; cursor: pointer; transition: all 0.15s; white-space: nowrap; }
.abtn:hover { background: #262b36; border-color: #7dd3fc; transform: translateY(-1px); }
.abtn.primary { background: #2563eb; border-color: #2563eb; color: white; }
.abtn.primary:hover { background: #1d4ed8; }
.abtn.purple { background: #7c3aed; border-color: #7c3aed; color: white; }
.abtn.purple:hover { background: #6d28d9; }
.abtn.danger { background: #7f1d1d; border-color: #991b1b; color: #fca5a5; }
.abtn.danger:hover { background: #991b1b; color: white; }
.abtn .ico { font-size: 16px; }
.abtn:disabled { opacity: 0.5; cursor: not-allowed; }

.model-selector { background: #171a21; border-bottom: 1px solid #262b36;
                  padding: 8px 16px; display: flex; gap: 8px; align-items: center; }
.model-selector select { flex: 1; max-width: 500px; background: #0f1116;
                         border: 1px solid #262b36; border-radius: 6px;
                         color: #e6e8ef; padding: 8px 12px; font-size: 13px;
                         font-family: inherit; outline: none; }
.model-selector select:focus { border-color: #2563eb; }
.ver { font-size: 11px; color: #4b5563; font-weight: 600;
       padding: 2px 8px; border: 1px solid #262b36;
       border-radius: 4px; background: #0f1116; }

.attachments {
  padding: 10px 16px; background: #13161d;
  border-bottom: 1px solid #262b36;
  display: none; gap: 10px; flex-wrap: wrap; align-items: center;
}
.attachments.active { display: flex; }
.attachments .label { font-size: 11px; color: #6b7280; }
.attach-chip {
  display: flex; align-items: center; gap: 8px;
  background: #1a1f2a; border: 1px solid #262b36;
  border-radius: 8px; padding: 6px 10px;
  font-size: 12px; color: #9ca3af; max-width: 280px;
  transition: border-color 0.15s;
}
.attach-chip:hover { border-color: #7dd3fc; }
.attach-chip img {
  width: 48px; height: 48px; object-fit: cover;
  border-radius: 6px; border: 1px solid #262b36;
}
.attach-chip .fname {
  overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; max-width: 140px;
}
.attach-chip .remove {
  cursor: pointer; color: #6b7280;
  font-weight: bold; padding: 2px 6px; font-size: 16px;
  border-radius: 4px; transition: all 0.15s;
}
.attach-chip .remove:hover { color: #f87171; background: #2a1616; }

.chat { flex: 1; overflow-y: auto; padding: 24px;
        display: flex; flex-direction: column; gap: 16px; }
.msg { max-width: 85%; padding: 12px 16px; border-radius: 12px;
       line-height: 1.65; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; }
.msg.user { align-self: flex-end; background: #2563eb; color: white;
            border-bottom-right-radius: 4px; }
.msg.system { align-self: center; background: transparent; color: #6b7280;
              font-size: 12px; padding: 4px 12px; max-width: none;
              font-style: italic; white-space: pre-wrap; }
.msg.error { color: #f87171; font-family: monospace; font-size: 12px; }
.msg.chat-image {
  align-self: flex-end; background: transparent; padding: 0; max-width: 300px;
}
.msg.chat-image img {
  max-width: 100%; border-radius: 12px; border: 2px solid #2563eb;
}

.assistant-wrap { align-self: flex-start; max-width: 85%;
                  display: flex; flex-direction: column; gap: 8px; }
.think-block { background: linear-gradient(135deg, #1a1f2a 0%, #1d2330 100%);
               border: 1px solid #2d3542; border-left: 3px solid #7dd3fc;
               border-radius: 8px; padding: 10px 14px;
               font-size: 13px; color: #9ca3af;
               font-family: "Consolas", "Monaco", monospace;
               line-height: 1.6; transition: all 0.25s ease;
               overflow: hidden; position: relative; }
.think-block .think-header { font-size: 12px;
                             font-family: -apple-system, "Segoe UI", sans-serif;
                             color: #7dd3fc; font-weight: 600;
                             margin-bottom: 6px; cursor: pointer;
                             user-select: none; display: flex;
                             align-items: center; gap: 6px; }
.think-block .think-header .arrow { display: inline-block;
                                    transition: transform 0.2s;
                                    font-size: 10px; color: #6b7280; }
.think-block.collapsed .think-header .arrow { transform: rotate(-90deg); }
.think-block .think-content { white-space: pre-wrap; word-wrap: break-word;
                              max-height: 400px; overflow-y: auto;
                              transition: max-height 0.3s ease, opacity 0.2s ease; opacity: 1; }
.think-block.collapsed .think-content { max-height: 0; opacity: 0; margin-top: -6px; }
.think-block.streaming .think-header .spinner {
  display: inline-block; width: 10px; height: 10px;
  border: 2px solid #7dd3fc; border-top-color: transparent;
  border-radius: 50%; animation: spin 0.7s linear infinite; }
.think-block:not(.streaming) .think-header .spinner { display: none; }
@keyframes spin { to { transform: rotate(360deg); } }
.answer-body { background: #1a1f2a; border: 1px solid #262b36;
               border-radius: 12px; border-bottom-left-radius: 4px;
               padding: 12px 16px; white-space: pre-wrap;
               word-wrap: break-word; font-size: 14px; line-height: 1.65; }
.answer-body.cursor::after { content: "▊"; color: #7dd3fc;
                              animation: blink 1s infinite; margin-left: 2px; }
@keyframes blink { 50% { opacity: 0; } }
.input-area { padding: 16px 24px 20px; border-top: 1px solid #262b36;
              background: #171a21; display: flex; gap: 10px; align-items: flex-end; }
.input-area textarea { flex: 1; background: #0f1116; border: 1px solid #262b36;
                       border-radius: 8px; color: #e6e8ef;
                       padding: 10px 14px; font-family: inherit; font-size: 14px;
                       resize: none; outline: none; min-height: 44px;
                       max-height: 180px; line-height: 1.5; }
.input-area textarea:focus { border-color: #2563eb; }
.input-area button { background: #2563eb; color: white; border: none;
                     border-radius: 8px; padding: 12px 22px;
                     font-size: 14px; font-weight: 600; cursor: pointer; height: 44px; }
.input-area button:disabled { background: #374151; cursor: not-allowed; }
.input-area button:hover:not(:disabled) { background: #1d4ed8; }
</style>
</head>
<body>

<div class="sidebar">
  <h1>💬 离线 AI v1.5</h1>
  <div>
    <h2>模型状态</h2>
    <div class="status-box" id="status">未加载模型</div>
  </div>
  <div>
    <h2>扫描路径</h2>
    <div class="scan-input-row">
      <input id="scanPath" value="F:\models">
      <button id="scanBtn" onclick="doScan()">扫描</button>
    </div>
  </div>
  <div>
    <h2>模型列表 <span id="modelCount" style="color:#4b5563;font-weight:400;"></span></h2>
    <div class="model-list" id="modelList">
      <div style="color:#6b7280;font-size:12px;padding:8px 0;">点顶部「🔍 扫描模型」</div>
    </div>
    <button class="btn ghost small" style="width:100%;margin-top:6px;" onclick="clearModels()">清空列表</button>
  </div>
  <div>
    <h2>图片 <span id="imgCount" style="color:#4b5563;font-weight:400;"></span></h2>
    <div id="imgList" style="font-size:12px;color:#6b7280;">暂无</div>
    <button class="btn ghost small" style="width:100%;margin-top:6px;" onclick="clearImages()">清空图片</button>
  </div>
  <div class="params">
    <h2>参数</h2>
    <label>温度 <span class="value" id="tempVal">0.7</span></label>
    <input type="range" id="temp" min="0" max="2" step="0.1" value="0.7"
           oninput="document.getElementById('tempVal').textContent=this.value">
    <label>最大 Token <span class="value" id="tokensVal">1024</span></label>
    <input type="range" id="tokens" min="64" max="4096" step="64" value="1024"
           oninput="document.getElementById('tokensVal').textContent=this.value">
  </div>
  <div style="margin-top:auto;font-size:11px;color:#4b5563;text-align:center;padding-top:12px;">
    ✓ 完全离线 · 不调用 API<br>
    GGUF: <span id="hasLlama">?</span> · PIL: <span id="hasPIL">?</span>
  </div>
</div>

<div class="main">
  <div class="action-bar">
    <span class="ver">v1.5</span>
    <button class="abtn primary" onclick="toolbarScan()"><span class="ico">🔍</span> 扫描模型</button>
    <button class="abtn primary" onclick="toolbarLoad()"><span class="ico">📂</span> 加载模型</button>
    <button class="abtn danger" onclick="toolbarUnload()"><span class="ico">⏏</span> 卸载</button>
    <span class="spacer"></span>
    <button class="abtn purple" onclick="document.getElementById('imgInput').click()"><span class="ico">🖼</span> 上传图片</button>
    <button class="abtn danger" onclick="toolbarClear()"><span class="ico">🗑</span> 清空对话</button>
    <input type="file" id="imgInput" style="display:none" multiple
           accept="image/*" onchange="handleImageUpload(this)">
  </div>

  <div class="model-selector">
    <select id="modelSelect">
      <option value="">-- 选择模型 --</option>
    </select>
    <button class="abtn primary" onclick="loadSelected()">📂 加载选中</button>
  </div>

  <div class="attachments" id="attachments">
    <span class="label">已上传：</span>
  </div>

  <div class="chat" id="chat">
    <div class="msg system">💡 点顶部「🔍 扫描模型」找模型 → 加载；或「🖼 上传图片」预览</div>
  </div>

  <div class="input-area">
    <textarea id="input" placeholder="输入消息… (Enter 发送，Shift+Enter 换行；/help 查看命令)"
              onkeydown="handleKey(event)" rows="1"></textarea>
    <button id="sendBtn" onclick="sendMessage()">发送</button>
  </div>
</div>

<script>
const chatEl=document.getElementById('chat');
const inputEl=document.getElementById('input');
const sendBtn=document.getElementById('sendBtn');
const modelSelect=document.getElementById('modelSelect');
let streaming=false;

function addMessage(role,text){
  const d=document.createElement('div');
  d.className='msg '+role; d.textContent=text;
  chatEl.appendChild(d); chatEl.scrollTop=chatEl.scrollHeight; return d;
}

function addImageToChat(name, thumbB64){
  const d=document.createElement('div');
  d.className='msg chat-image';
  d.innerHTML=`<img src="data:image/jpeg;base64,${thumbB64}" title="${name}">`;
  chatEl.appendChild(d); chatEl.scrollTop=chatEl.scrollHeight;
}

function handleKey(e){
  if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendMessage(); }
}

async function handleImageUpload(input){
  const files=input.files;
  if(!files||files.length===0) return;
  for(const f of files){
    const fd=new FormData(); fd.append('file',f);
    addMessage('system',`📤 上传图片：${f.name} (${(f.size/1024).toFixed(1)} KB)`);
    try {
      const r=await fetch('/api/upload_image',{method:'POST',body:fd});
      const d=await r.json();
      if(d.ok){
        addMessage('system',`✓ ${f.name} 已上传`);
        addImageToChat(f.name, d.thumb);
      } else {
        addMessage('system',`✗ ${f.name}：${d.msg}`);
      }
    } catch(e){
      addMessage('system',`✗ ${f.name} 上传失败：${e.message}`);
    }
  }
  input.value='';
  refreshImages();
}

async function refreshImages(){
  try {
    const r=await fetch('/api/images'); const d=await r.json();
    const list=d.images||[];
    const el=document.getElementById('attachments');
    const sb=document.getElementById('imgList');
    const cnt=document.getElementById('imgCount');
    cnt.textContent=list.length?`(${list.length})`:'';
    if(list.length===0){
      el.classList.remove('active');
      el.innerHTML='<span class="label">已上传：</span>';
      sb.textContent='暂无';
      return;
    }
    el.classList.add('active');
    el.innerHTML='<span class="label">已上传：</span>';
    list.forEach(u=>{
      const c=document.createElement('div'); c.className='attach-chip';
      c.innerHTML=`<img src="data:image/jpeg;base64,${u.thumb}">
                   <span class="fname" title="${u.name}">${u.name}</span>
                   <span class="remove" onclick="removeImage('${u.id}')">×</span>`;
      el.appendChild(c);
    });
    sb.innerHTML='';
    list.forEach(u=>{
      const l=document.createElement('div');
      l.style.cssText='padding:3px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
      l.title=u.name; l.textContent=`🖼 ${u.name}`;
      sb.appendChild(l);
    });
  } catch(e){}
}

async function removeImage(id){
  await fetch('/api/images/'+id,{method:'DELETE'});
  refreshImages();
}
async function clearImages(){
  await fetch('/api/images',{method:'DELETE'});
  refreshImages();
}

async function toolbarScan(){
  const p=prompt("输入要扫描的路径：","F:\\models");
  if(!p) return;
  document.getElementById('scanPath').value=p;
  await doScan();
}
async function toolbarLoad(){
  const path=modelSelect.value;
  if(!path){ const p=prompt("输入模型路径：","F:\\models\\"); if(!p) return; await loadModel(p,''); return; }
  const opt=modelSelect.options[modelSelect.selectedIndex];
  await loadModel(path, opt.dataset.type||'');
}
async function loadSelected(){ await toolbarLoad(); }
async function toolbarUnload(){
  await fetch('/api/unload',{method:'POST'});
  addMessage('system','✓ 模型已卸载'); loadStatus(); loadModels();
}
async function toolbarClear(){
  await fetch('/api/clear',{method:'POST'});
  chatEl.innerHTML='<div class="msg system">对话已清空</div>';
}

function createAssistantWrapper(){
  const w=document.createElement('div'); w.className='assistant-wrap';
  chatEl.appendChild(w); return w;
}
function createThinkBlock(w){
  const b=document.createElement('div'); b.className='think-block streaming';
  b.innerHTML=`<div class="think-header"><span class="arrow">▼</span><span class="spinner"></span><span class="label">💭 正在思考…</span></div><div class="think-content"></div>`;
  b.querySelector('.think-header').addEventListener('click',()=>b.classList.toggle('collapsed'));
  w.appendChild(b);
  return {block:b,content:b.querySelector('.think-content'),label:b.querySelector('.label')};
}
function finishThink(t){
  if(!t) return; t.block.classList.remove('streaming');
  t.block.classList.add('collapsed'); t.label.textContent='✓ 已思考（点击展开）';
}
function getBody(w){
  let b=w.querySelector('.answer-body');
  if(!b){ b=document.createElement('div'); b.className='answer-body cursor'; w.appendChild(b); }
  return b;
}
async function sendMessage(){
  if(streaming) return;
  const text=inputEl.value.trim(); if(!text) return;
  inputEl.value=''; addMessage('user',text);
  const w=createAssistantWrapper();
  let tr=null, ab=null, tf=false;
  streaming=true; sendBtn.disabled=true;
  const fin=()=>{ if(tr&&!tf){ finishThink(tr); tf=true; } };
  try {
    const resp=await fetch('/api/chat',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text,temperature:parseFloat(document.getElementById('temp').value),
        max_new_tokens:parseInt(document.getElementById('tokens').value)})});
    const rd=resp.body.getReader(); const dc=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await rd.read(); if(done) break;
      buf+=dc.decode(value,{stream:true});
      const ps=buf.split('\n\n'); buf=ps.pop();
      for(const p of ps){
        if(!p.startsWith('data: ')) continue;
        try {
          const d=JSON.parse(p.slice(6));
          if(d.type==='think'){
            if(!tr) tr=createThinkBlock(w);
            tr.content.textContent+=d.content; chatEl.scrollTop=chatEl.scrollHeight;
          } else if(d.type==='text'){
            fin(); ab=getBody(w); ab.textContent+=d.content; chatEl.scrollTop=chatEl.scrollHeight;
          } else if(d.type==='error'){
            fin(); const ed=document.createElement('div'); ed.className='msg error';
            ed.textContent='错误：'+d.content; w.appendChild(ed);
          }
        } catch(e){}
      }
    }
    fin(); if(ab) ab.classList.remove('cursor');
    if(!ab&&!tr){ const e=document.createElement('div'); e.className='answer-body';
      e.textContent='（无输出）'; w.appendChild(e); }
  } catch(e){
    fin(); const ed=document.createElement('div'); ed.className='msg error';
    ed.textContent='网络错误：'+e.message; w.appendChild(ed);
  } finally {
    streaming=false; sendBtn.disabled=false; inputEl.focus();
    loadStatus(); loadModels();
  }
}

async function loadStatus(){
  try {
    const r=await fetch('/api/status'); const d=await r.json();
    const el=document.getElementById('status');
    if(d.backend) el.innerHTML=`<span class="ok">● 已加载</span><br><b>${d.name}</b><br><span style="font-size:11px;">${d.path}</span>`;
    else el.innerHTML='<span class="err">○ 未加载</span>';
    document.getElementById('hasLlama').textContent=d.has_llama?'✓':'✗';
    document.getElementById('hasPIL').textContent=d.has_pil?'✓':'✗';
  } catch(e){}
}
async function loadModels(){
  try {
    const r=await fetch('/api/models'); const d=await r.json();
    const el=document.getElementById('modelList');
    const cnt=document.getElementById('modelCount');
    const cv=modelSelect.value;
    modelSelect.innerHTML='<option value="">-- 选择模型 --</option>';
    if(d.models&&d.models.length>0){
      d.models.forEach(m=>{
        const o=document.createElement('option'); o.value=m.path;
        o.textContent=`[${m.type}] ${m.name}${m.active?' (当前)':''}`;
        o.dataset.type=m.type; modelSelect.appendChild(o);
      });
      if(cv) modelSelect.value=cv;
    }
    if(!d.models||d.models.length===0){
      el.innerHTML='<div style="color:#6b7280;font-size:12px;padding:8px 0;">点顶部「🔍 扫描模型」</div>';
      cnt.textContent=''; return;
    }
    cnt.textContent='('+d.models.length+')'; el.innerHTML='';
    d.models.forEach(m=>{
      const div=document.createElement('div');
      div.className='model-item'+(m.active?' active':'');
      div.title=m.path;
      const sn=m.name.length>28?m.name.slice(0,26)+'…':m.name;
      div.innerHTML=`<span class="name">${sn}</span><span class="type">${m.type}</span>`;
      div.onclick=()=>loadModel(m.path,m.type); el.appendChild(div);
    });
  } catch(e){}
}
async function loadModel(path,type){
  addMessage('system','正在加载 '+path+' …');
  try {
    const r=await fetch('/api/load',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path,type})});
    const d=await r.json();
    addMessage('system',(d.ok?'✓ ':'✗ ')+d.msg); loadStatus(); loadModels();
  } catch(e){ addMessage('system','加载失败：'+e.message); }
}
async function doScan(){
  const path=document.getElementById('scanPath').value.trim();
  if(!path) return;
  const btn=document.getElementById('scanBtn');
  btn.disabled=true; btn.textContent='扫描中…';
  addMessage('system','正在扫描 '+path+' …');
  try {
    const r=await fetch('/api/scan',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path,max_depth:8,timeout:60})});
    const d=await r.json();
    addMessage('system',(d.ok?'✓ ':'✗ ')+d.msg); loadModels();
  } catch(e){ addMessage('system','扫描失败：'+e.message); }
  finally { btn.disabled=false; btn.textContent='扫描'; }
}
async function clearModels(){ await fetch('/api/clear_models',{method:'POST'}); loadModels(); }

loadStatus(); loadModels(); refreshImages(); inputEl.focus();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    resp = Response(HTML_PAGE, mimetype='text/html')
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/api/status')
def api_status():
    return jsonify({"backend": llm.backend, "name": llm.current_name or "",
                    "path": llm.model_path or "", "has_llama": HAS_LLAMA,
                    "has_pil": HAS_PIL})


@app.route('/api/models')
def api_models():
    return jsonify({"models": [{**m, "active": m["path"] == llm.model_path}
                               for m in _list_models()]})


@app.route('/api/images')
def api_images():
    return jsonify({"images": [{"id": u["id"], "name": u["name"],
                                "thumb": u["thumb"], "size": u["size"]}
                               for u in _list_images()]})


@app.route('/api/upload_image', methods=['POST'])
def api_upload_image():
    if 'file' not in request.files:
        return jsonify({"ok": False, "msg": "没有文件"})
    f = request.files['file']
    if not f.filename:
        return jsonify({"ok": False, "msg": "文件名为空"})
    try:
        data = f.read()
        name = f.filename
        fid = f"i{int(time.time()*1000)}"
        thumb = make_thumb(data)
        _add_image({"id": fid, "name": name, "thumb": thumb, "size": len(data)})
        return jsonify({"ok": True, "id": fid, "thumb": thumb,
                        "name": name, "size": len(data)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "msg": str(e)})


@app.route('/api/images/<file_id>', methods=['DELETE'])
def api_rm_image(file_id):
    return jsonify({"ok": _rm_image(file_id)})


@app.route('/api/images', methods=['DELETE'])
def api_cl_images():
    _clear_images()
    return jsonify({"ok": True})


@app.route('/api/scan', methods=['POST'])
def api_scan():
    d = request.json or {}
    p = (d.get('path') or '').strip().strip('"')
    if not p:
        return jsonify({"ok": False, "msg": "路径为空"})
    if not os.path.isdir(p):
        return jsonify({"ok": False, "msg": f"不是目录：{p}"})
    try:
        f = scan_models(p, max_depth=int(d.get('max_depth', 8)),
                        timeout=int(d.get('timeout', 60)))
    except Exception as e:
        return jsonify({"ok": False, "msg": f"扫描出错：{e}"})
    n = sum(1 for m in f if _add_model(m["name"], m["path"], m["type"]))
    return jsonify({"ok": True, "msg": f"在 {p} 找到 {len(f)} 个模型，新增 {n} 个"})


@app.route('/api/clear_models', methods=['POST'])
def api_cl_models():
    _clear_models()
    return jsonify({"ok": True})


@app.route('/api/load', methods=['POST'])
def api_load():
    d = request.json or {}
    p = (d.get('path') or '').strip().strip('"')
    t = d.get('type', 'gguf')
    if not p:
        return jsonify({"ok": False, "msg": "路径为空"})
    if p.lower().endswith('.gguf'):
        t = 'gguf'
    elif os.path.isdir(p):
        if any(f.lower().endswith(".gguf") for f in os.listdir(p)):
            t = 'gguf'
    ok, msg = llm.load_gguf(p)
    if ok:
        llm.current_name = os.path.basename(p.rstrip("/\\"))
        _add_model(llm.current_name, p, t)
    return jsonify({"ok": ok, "msg": msg})


@app.route('/api/unload', methods=['POST'])
def api_unload():
    llm.unload()
    return jsonify({"ok": True})


@app.route('/api/clear', methods=['POST'])
def api_clear():
    conversation_history.clear()
    return jsonify({"ok": True})


@app.route('/api/chat', methods=['POST'])
def api_chat():
    d = request.json or {}
    ut = (d.get('text') or '').strip()
    if not ut:
        return jsonify({"error": "empty"}), 400
    temp = float(d.get('temperature', 0.7))
    mx = int(d.get('max_new_tokens', 1024))

    def gen():
        c = parse_cmd(ut)
        if c:
            cmd, args = c
            try:
                res = exec_cmd(cmd, args)
            except Exception as e:
                traceback.print_exc()
                res = f"✗ {e}"
            yield f"data: {json.dumps({'type': 'text', 'content': res}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        imgs = _list_images()
        img_note = ""
        if imgs:
            img_note = f"\n\n[用户附带了 {len(imgs)} 张图片：{', '.join(u['name'] for u in imgs)}。当前模型不支持读图，如需理解图片内容请告诉我图片里是什么。]"

        full_msg = ut + img_note

        conversation_history.append(("user", ut))
        if len(conversation_history) > MAX_TURNS * 2:
            del conversation_history[:len(conversation_history) - MAX_TURNS * 2]

        q = try_math(ut)
        if q and not imgs:
            conversation_history.append(("assistant", q))
            yield f"data: {json.dumps({'type': 'text', 'content': q}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        if not llm.backend:
            rep = "（未加载模型。点顶部「📂 加载模型」按钮。）"
            conversation_history.append(("assistant", rep))
            yield f"data: {json.dumps({'type': 'text', 'content': rep}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        temp_hist = conversation_history[:-1] + [("user", full_msg)]
        collected = []
        try:
            ri = llm.chat_stream(temp_hist, mx=mx, temp=temp)
            for kind, piece in stream_think_filter(ri):
                if not piece:
                    continue
                if kind == "text":
                    collected.append(piece)
                yield f"data: {json.dumps({'type': kind, 'content': piece}, ensure_ascii=False)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
            return
        rep = "".join(collected).strip()
        conversation_history.append(("assistant", rep))
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(gen(), mimetype='text/event-stream',
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}"
    print("=" * 60)
    print("离线 AI Web 版 v1.5")
    print(f"GGUF: {'✓' if HAS_LLAMA else '✗'}  PIL: {'✓' if HAS_PIL else '✗'}  TF: ✗（已禁用）")
    print(f"浏览器访问：{url}")
    print("=" * 60)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
