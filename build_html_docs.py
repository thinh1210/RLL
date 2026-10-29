#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_html_docs.py
==================
Trình biên dịch toàn bộ hệ thống tài liệu Markdown (.md) của kho lưu trữ
Reinforcement-Learning sang hệ thống trang web HTML tĩnh (Static Local Web Portal)
để đọc trực tiếp trên trình duyệt local (file:// hoặc http://localhost).

Tính năng nổi bật:
1. Bảo vệ tuyệt đối các khối công thức Toán học LaTeX ($$...$$ và $...$) trước khi
   chạy Markdown parser (tránh lỗi ký tự '_', '*', '|' trong bảng làm vỡ công thức).
2. Hỗ trợ đầy đủ các lệnh LaTeX nâng cao: \\boxed, \\cancel, \\underbrace, \\begin{aligned}...
   thông qua MathJax 3 (kèm extension 'cancel', 'ams').
3. Tự động chuyển đổi các liên kết nội bộ (.md và file:///.../Reinforcement-Learning/...)
   sang liên kết tương đối (.html) để bấm chuyển trang mượt mà trên trình duyệt.
4. Giao diện hiện đại (Sáng/Tối), Thanh điều hướng 21 tài liệu bên trái, Mục lục tự động
   (Table of Contents) bên phải, Tìm kiếm nhanh và Nút chuyển bài Trước/Sau.
"""

import html
import os
import re
from pathlib import Path
import markdown
from interactive_labs import get_interactive_lab_html

ROOT_DIR = Path(__file__).resolve().parent

# Danh sách 21 tài liệu chuẩn theo đúng thứ tự lộ trình học
DOCS_MANIFEST = [
    # Nhóm 0: Trang chủ
    {
        "id": "home",
        "group": "🏠 Tổng Quan Hệ Thống",
        "title": "Trang Chủ & Bản Đồ Toàn Khóa",
        "md_path": "README.md",
        "html_path": "index.html",
        "extra_html_path": "README.html",
        "badge": "Portal",
    },
    # Nhóm 1: Nền tảng Toán học & Cẩm nang Tham số
    {
        "id": "math_symbols",
        "group": "📐 Toán Học & Cẩm Nang Tham Số",
        "title": "1. Bảng Ký Hiệu Toán Học Chuẩn Tắc",
        "md_path": "BANG_KY_HIEU_TOAN_HOC.md",
        "html_path": "BANG_KY_HIEU_TOAN_HOC.html",
        "badge": "Ký hiệu",
    },
    {
        "id": "math_foundations",
        "group": "📐 Toán Học & Cẩm Nang Tham Số",
        "title": "2. Tổng Hợp XSTK, ĐSTT & Các Bước Thế Hàm",
        "md_path": "TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.md",
        "html_path": "TONG_HOP_TOAN_HOC_XSTK_DSTT_RL.html",
        "badge": "Toán cốt lõi",
    },
    {
        "id": "param_handbook",
        "group": "📐 Toán Học & Cẩm Nang Tham Số",
        "title": "3. Cẩm Nang Toàn Diện Về Tham Số RL",
        "md_path": "CAM_NANG_THAM_SO_RL.md",
        "html_path": "CAM_NANG_THAM_SO_RL.html",
        "badge": "Tham số",
    },
    {
        "id": "prereq_roadmap",
        "group": "📐 Toán Học & Cẩm Nang Tham Số",
        "title": "4. Lộ Trình Kiến Thức Cần Học",
        "md_path": "KIEN_THUC_CAN_HOC.md",
        "html_path": "KIEN_THUC_CAN_HOC.html",
        "badge": "Lộ trình",
    },
    {
        "id": "weekly_summary",
        "group": "📐 Toán Học & Cẩm Nang Tham Số",
        "title": "5. Tổng Hợp Kiến Thức & Đối Chiếu Code 8 Tuần",
        "md_path": "KIEN_THUC_CHI_TIET_TUNG_TUAN.md",
        "html_path": "KIEN_THUC_CHI_TIET_TUNG_TUAN.html",
        "badge": "Tổng hợp",
    },
    {
        "id": "env_guide",
        "group": "📐 Toán Học & Cẩm Nang Tham Số",
        "title": "6. Hướng Dẫn Môi Trường (.venv), Chạy Code & ROS 2",
        "md_path": "HUONG_DAN_MOI_TRUONG_VA_CHAY_CODE.md",
        "html_path": "HUONG_DAN_MOI_TRUONG_VA_CHAY_CODE.html",
        "badge": "Môi trường",
    },
    # Nhóm 2: Bài giảng chuyên sâu 8 tuần
    {
        "id": "week1",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 1: Giới Thiệu RL, Bandit (UCB1) & Pong",
        "md_path": "Week1/BAI_GIANG.md",
        "html_path": "Week1/BAI_GIANG.html",
        "badge": "Tuần 1",
    },
    {
        "id": "week2",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 2: MDP, Bellman, Co Banach & Q-Learning",
        "md_path": "Week2/BAI_GIANG.md",
        "html_path": "Week2/BAI_GIANG.html",
        "badge": "Tuần 2",
    },
    {
        "id": "week3",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 3: Deep Q-Networks (DQN, Double, Dueling)",
        "md_path": "Week3/BAI_GIANG.md",
        "html_path": "Week3/BAI_GIANG.html",
        "badge": "Tuần 3",
    },
    {
        "id": "week4",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 4: Policy Gradient, REINFORCE & A2C",
        "md_path": "Week4/BAI_GIANG.md",
        "html_path": "Week4/BAI_GIANG.html",
        "badge": "Tuần 4",
    },
    {
        "id": "week5",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 5: TRPO, PPO-Clip & Chuỗi GAE",
        "md_path": "Week5/BAI_GIANG.md",
        "html_path": "Week5/BAI_GIANG.html",
        "badge": "Tuần 5",
    },
    {
        "id": "week6",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 6: Evolution Strategies (ES) & GA",
        "md_path": "Week6/BAI_GIANG.md",
        "html_path": "Week6/BAI_GIANG.html",
        "badge": "Tuần 6",
    },
    {
        "id": "week7",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 7: Model-Based RL, Tikhonov & MPC",
        "md_path": "Week7/BAI_GIANG.md",
        "html_path": "Week7/BAI_GIANG.html",
        "badge": "Tuần 7",
    },
    {
        "id": "week8",
        "group": "🎓 Bài Giảng Chuyên Sâu 8 Tuần",
        "title": "Week 8: SAC, IRL, RND, AlphaZero, RLHF & DPO",
        "md_path": "Week8/BAI_GIANG.md",
        "html_path": "Week8/BAI_GIANG.html",
        "badge": "Tuần 8",
    },
    # Nhóm 3: Simulation Vật lý & Sim-to-Real Robotics
    {
        "id": "sim_guide",
        "group": "🤖 Simulation & Sim-to-Real Robotics",
        "title": "Cẩm Nang Simulation & Cấu Hình Vật Lý Robot",
        "md_path": "HUONG_DAN_SIMULATION_VA_VAT_LY_ROBOT.md",
        "html_path": "HUONG_DAN_SIMULATION_VA_VAT_LY_ROBOT.html",
        "badge": "Simulation",
    },
    {
        "id": "sim2real_overview",
        "group": "🤖 Simulation & Sim-to-Real Robotics",
        "title": "Tổng Quan Kiến Trúc 7 Tầng Sim-to-Real",
        "md_path": "Sim2Real_Robotics/README.md",
        "html_path": "Sim2Real_Robotics/README.html",
        "badge": "7 Tầng",
    },
    {
        "id": "sim2real_p1",
        "group": "🤖 Simulation & Sim-to-Real Robotics",
        "title": "Phần 1 (Tầng 1): Mô Tả Cơ Học URDF/MJCF",
        "md_path": "Sim2Real_Robotics/Part1_Physical_Modeling_URDF_MJCF.md",
        "html_path": "Sim2Real_Robotics/Part1_Physical_Modeling_URDF_MJCF.html",
        "badge": "Tầng 1",
    },
    {
        "id": "sim2real_p2",
        "group": "🤖 Simulation & Sim-to-Real Robotics",
        "title": "Phần 2 (Tầng 2): Bộ Giải Động Lực Học & Cảm Biến",
        "md_path": "Sim2Real_Robotics/Part2_Physics_Engine_And_Sensors.md",
        "html_path": "Sim2Real_Robotics/Part2_Physics_Engine_And_Sensors.html",
        "badge": "Tầng 2",
    },
    {
        "id": "sim2real_p3",
        "group": "🤖 Simulation & Sim-to-Real Robotics",
        "title": "Phần 3 (Tầng 3): Môi Trường RL & Domain Randomization",
        "md_path": "Sim2Real_Robotics/Part3_RL_Training_Environment_And_Rewards.md",
        "html_path": "Sim2Real_Robotics/Part3_RL_Training_Environment_And_Rewards.html",
        "badge": "Tầng 3",
    },
    {
        "id": "sim2real_p4",
        "group": "🤖 Simulation & Sim-to-Real Robotics",
        "title": "Phần 4 (Tầng 4): 7 Bài Kiểm Định Vật Lý Bằng Code",
        "md_path": "Sim2Real_Robotics/Part4_Code_Only_Physical_Verification.md",
        "html_path": "Sim2Real_Robotics/Part4_Code_Only_Physical_Verification.html",
        "badge": "Tầng 4",
    },
    {
        "id": "sim2real_p5",
        "group": "🤖 Simulation & Sim-to-Real Robotics",
        "title": "Phần 5 (Tầng 5–7): Cầu Nối ROS 2 & Robot Thật",
        "md_path": "Sim2Real_Robotics/Part5_ROS2_Gazebo_And_Real_Robot_Bridge.md",
        "html_path": "Sim2Real_Robotics/Part5_ROS2_Gazebo_And_Real_Robot_Bridge.html",
        "badge": "Tầng 5-7",
    },
]


def ensure_list_spacing(md_text: str) -> str:
    """
    Đảm bảo có dòng trống trước danh sách (-, *, 1.) để Python-Markdown
    luôn sinh thẻ <ul>/<ol> chuẩn xác (kể cả bên trong blockquote '>').
    """
    lines = md_text.splitlines()
    out = []
    in_fence = False
    list_pattern = re.compile(r"^(\s*(?:>\s*)*)([-*+]|\d+\.)\s+")
    table_pattern = re.compile(r"^\s*(?:>\s*)*\|")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        m = list_pattern.match(line)
        if m and i > 0:
            prefix = m.group(1)
            prev = lines[i - 1]
            prev_stripped = prev.strip()
            # Nếu dòng trước không trống, không phải tiêu đề, không phải list item, không phải bảng
            if (
                prev_stripped != ""
                and prev_stripped != ">"
                and not list_pattern.match(prev)
                and not prev_stripped.startswith("#")
                and not table_pattern.match(prev)
            ):
                if prefix.strip().startswith(">"):
                    out.append(prefix.rstrip())
                else:
                    out.append("")
        out.append(line)
    return "\n".join(out)


def rewrite_links(md_text: str, current_md_rel: str) -> str:
    """
    Chuyển đổi các đường dẫn file:///home/thinh/Reinforcement-Learning/...
    và các đường dẫn .md nội bộ sang đường dẫn .html tương đối để duyệt web local.
    """
    current_dir = (ROOT_DIR / current_md_rel).parent

    # 1. Thay thế file:///home/thinh/Reinforcement-Learning/... bằng đường dẫn tương đối
    repo_file_prefix = f"file://{ROOT_DIR.as_posix()}/"

    def replace_repo_file_url(match):
        full_target = match.group(1)
        anchor = ""
        if "#" in full_target:
            target_path_str, anchor = full_target.split("#", 1)
            anchor = "#" + anchor
        else:
            target_path_str = full_target

        abs_target = ROOT_DIR / target_path_str
        try:
            rel_path = os.path.relpath(abs_target, current_dir).replace("\\", "/")
        except ValueError:
            rel_path = target_path_str

        if rel_path.endswith(".md"):
            rel_path = rel_path[:-3] + ".html"
        return f"]({rel_path}{anchor})"

    md_text = re.sub(
        r"\]\(file://" + re.escape(ROOT_DIR.as_posix()) + r"/([^)]+)\)",
        replace_repo_file_url,
        md_text,
    )

    # 2. Thay thế các đường dẫn tương đối kết thúc bằng .md (hoặc .md#anchor)
    def replace_md_link(match):
        url = match.group(1).strip()
        if url.startswith(("http://", "https://", "mailto:", "file://", "#")):
            return match.group(0)
        anchor = ""
        if "#" in url:
            path_part, anchor = url.split("#", 1)
            anchor = "#" + anchor
        else:
            path_part = url
        if path_part.endswith(".md"):
            path_part = path_part[:-3] + ".html"
        return f"]({path_part}{anchor})"

    md_text = re.sub(r"\]\(([^)]+)\)", replace_md_link, md_text)
    return md_text


def convert_markdown_with_math(md_text: str, current_md_rel: str) -> str:
    """
    Chuyển đổi Markdown sang HTML đồng thời bảo vệ nguyên vẹn Code Blocks và Công thức Toán LaTeX.
    """
    md_text = rewrite_links(md_text, current_md_rel)

    code_store = {}
    math_store = {}
    code_counter = 0
    math_counter = 0

    # Bước 1: Bảo vệ Fenced Code Blocks (```...``` hoặc ````...````)
    def save_fenced_code(match):
        nonlocal code_counter
        key = f"FENCEDCODEPLACEHOLDER{code_counter}END"
        code_counter += 1
        code_store[key] = match.group(0)
        return key

    md_text = re.sub(
        r"(?m)^([ \t>]*`{3,}.*?^[ \t>]*`{3,})",
        save_fenced_code,
        md_text,
        flags=re.DOTALL,
    )

    # Bước 2: Bảo vệ Inline Code (`...`)
    def save_inline_code(match):
        nonlocal code_counter
        key = f"INLINECODEPLACEHOLDER{code_counter}END"
        code_counter += 1
        code_store[key] = match.group(0)
        return key

    md_text = re.sub(r"`[^`\n]+`", save_inline_code, md_text)

    # Bước 3: Bảo vệ Display Math ($$...$$)
    def save_display_math(match):
        nonlocal math_counter
        key = f"DISPLAYMATHPLACEHOLDER{math_counter}END"
        math_counter += 1
        raw_math = match.group(1)
        # Nếu khối $$...$$ nằm bên trong blockquote (> ...), xóa ký tự '>' ở đầu các dòng tiếp theo
        cleaned_lines = [
            re.sub(r"^[ \t]*>[ \t]?", "", line) for line in raw_math.splitlines()
        ]
        cleaned_math = "\n".join(cleaned_lines)
        math_store[key] = f'<div class="math-block">$${html.escape(cleaned_math)}$$</div>'
        return key

    md_text = re.sub(r"\$\$(.+?)\$\$", save_display_math, md_text, flags=re.DOTALL)

    # Bước 4: Bảo vệ Inline Math ($...$)
    def save_inline_math(match):
        nonlocal math_counter
        key = f"INLINEMATHPLACEHOLDER{math_counter}END"
        math_counter += 1
        raw_math = match.group(1)
        math_store[key] = f'<span class="math-inline">${html.escape(raw_math)}$</span>'
        return key

    md_text = re.sub(
        r"(?<!\\)\$(?!\s)([^\$\n]+?)(?<!\s)(?<!\\)\$",
        save_inline_math,
        md_text,
    )

    # Bước 5: Khôi phục Code Blocks trước khi chạy Markdown parser
    for key, val in code_store.items():
        md_text = md_text.replace(key, val)

    # Đảm bảo khoảng cách dòng trước danh sách
    md_text = ensure_list_spacing(md_text)

    # Bước 6: Chuyển đổi Markdown sang HTML
    md_parser = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        extension_configs={
            "toc": {
                "permalink": True,
                "permalink_title": "Liên kết tới mục này",
            }
        },
    )
    html_body = md_parser.convert(md_text)

    # Bước 7: Khôi phục các khối Toán học LaTeX
    for key, val in math_store.items():
        html_body = html_body.replace(key, val)

    # Bước 8: Nâng cấp giao diện cho các hộp Blockquote đặc biệt (Giải phẫu thế hàm, Bản chất Logic, GitHub Alerts)
    html_body = enhance_callouts(html_body)

    toc_html = build_toc_html(getattr(md_parser, "toc_tokens", []), math_store)
    return html_body, toc_html


def enhance_callouts(html_body: str) -> str:
    """
    Thêm class CSS chuyên biệt cho các hộp blockquote quan trọng:
    - 🔄 Giải phẫu các bước thế hàm -> .callout-substitution
    - 💡 Bản chất Logic / Mục đích ra đời -> .callout-logic
    - 🎯 Tác động lên mẫu / Ý nghĩa -> .callout-sample
    - [!NOTE], [!IMPORTANT], [!WARNING], [!TIP] -> .callout-alert
    """
    def classify_blockquote(match):
        inner = match.group(1)
        if "Giải phẫu các bước thế hàm" in inner or "🔄" in inner:
            return f'<blockquote class="callout callout-substitution">{inner}</blockquote>'
        if "Bản chất Logic" in inner or "💡" in inner:
            return f'<blockquote class="callout callout-logic">{inner}</blockquote>'
        if "Tác động lên mẫu" in inner or "🎯" in inner:
            return f'<blockquote class="callout callout-sample">{inner}</blockquote>'
        for tag, cls, label in [
            ("[!NOTE]", "callout-note", "📌 Ghi chú"),
            ("[!TIP]", "callout-tip", "💡 Mẹo thực chiến"),
            ("[!IMPORTANT]", "callout-important", "⚡ Quan trọng"),
            ("[!WARNING]", "callout-warning", "⚠️ Cảnh báo"),
            ("[!CAUTION]", "callout-warning", "🚨 Thận trọng"),
        ]:
            if tag in inner:
                cleaned = inner.replace(tag, f"<strong>{label}:</strong>", 1)
                return f'<blockquote class="callout {cls}">{cleaned}</blockquote>'
        return f'<blockquote class="callout">{inner}</blockquote>'

    return re.sub(
        r"<blockquote>(.*?)</blockquote>",
        classify_blockquote,
        html_body,
        flags=re.DOTALL,
    )


def build_sidebar_html(current_doc_id: str, current_html_rel: str) -> str:
    current_dir = (ROOT_DIR / current_html_rel).parent
    groups = {}
    for doc in DOCS_MANIFEST:
        groups.setdefault(doc["group"], []).append(doc)

    parts = []
    for group_name, docs in groups.items():
        parts.append(f'<div class="nav-group"><div class="nav-group-title">{html.escape(group_name)}</div>')
        for doc in docs:
            target_abs = ROOT_DIR / doc["html_path"]
            rel_link = os.path.relpath(target_abs, current_dir).replace("\\", "/")
            active_cls = " active" if doc["id"] == current_doc_id else ""
            badge_html = f'<span class="nav-badge">{html.escape(doc["badge"])}</span>' if doc.get("badge") else ""
            parts.append(
                f'<a class="nav-item{active_cls}" href="{rel_link}" data-title="{html.escape(doc["title"].lower())}">'
                f'<span class="nav-item-text">{html.escape(doc["title"])}</span>{badge_html}</a>'
            )
        parts.append("</div>")
    return "\n".join(parts)


def build_toc_html(toc_tokens: list, math_store: dict) -> str:
    if not toc_tokens:
        return '<p class="toc-empty">Trang này không có mục lục phụ.</p>'

    def restore_toc_name(name_str: str) -> str:
        # name_str từ markdown toc_tokens đã được escape HTML sẵn
        for k, v in math_store.items():
            if k in name_str:
                name_str = name_str.replace(k, v)
        return name_str

    items = []
    for token in toc_tokens:
        # Nếu có thẻ h1 duy nhất ở gốc, lấy các con h2/h3 của nó
        children = token.get("children", [])
        if token.get("level") == 1 and children:
            for h2 in children:
                items.append(
                    f'<a class="toc-link toc-l2" href="#{h2["id"]}">{restore_toc_name(h2["name"])}</a>'
                )
                for h3 in h2.get("children", []):
                    items.append(
                        f'<a class="toc-link toc-l3" href="#{h3["id"]}">{restore_toc_name(h3["name"])}</a>'
                    )
        else:
            lvl = min(max(token.get("level", 2), 2), 3)
            items.append(
                f'<a class="toc-link toc-l{lvl}" href="#{token["id"]}">{restore_toc_name(token["name"])}</a>'
            )
            for child in children:
                items.append(
                    f'<a class="toc-link toc-l3" href="#{child["id"]}">{restore_toc_name(child["name"])}</a>'
                )
    return "\n".join(items) if items else '<p class="toc-empty">Trang này không có mục lục phụ.</p>'


def build_prev_next_html(idx: int, current_html_rel: str) -> str:
    current_dir = (ROOT_DIR / current_html_rel).parent
    prev_html = '<div class="pager-spacer"></div>'
    next_html = '<div class="pager-spacer"></div>'

    if idx > 0:
        prev_doc = DOCS_MANIFEST[idx - 1]
        prev_rel = os.path.relpath(ROOT_DIR / prev_doc["html_path"], current_dir).replace("\\", "/")
        prev_html = (
            f'<a class="pager-card pager-prev" href="{prev_rel}">'
            f'<span class="pager-label">← Bài trước</span>'
            f'<span class="pager-title">{html.escape(prev_doc["title"])}</span>'
            f"</a>"
        )
    if idx < len(DOCS_MANIFEST) - 1:
        next_doc = DOCS_MANIFEST[idx + 1]
        next_rel = os.path.relpath(ROOT_DIR / next_doc["html_path"], current_dir).replace("\\", "/")
        next_html = (
            f'<a class="pager-card pager-next" href="{next_rel}">'
            f'<span class="pager-label">Bài tiếp theo →</span>'
            f'<span class="pager-title">{html.escape(next_doc["title"])}</span>'
            f"</a>"
        )
    return f'<nav class="doc-pager">{prev_html}{next_html}</nav>'


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{page_title} — Deep RL & Sim-to-Real Robotics</title>
  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
  <!-- Highlight.js for syntax highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <!-- MathJax 3 for full LaTeX support (\\boxed, \\cancel, \\underbrace, aligned) -->
  <script>
    window.MathJax = {{
      loader: {{ load: ['[tex]/cancel', '[tex]/ams', '[tex]/boldsymbol'] }},
      tex: {{
        packages: {{ '[+]': ['cancel', 'ams', 'boldsymbol'] }},
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root[data-theme="dark"] {{
      --bg-primary: #0d1117;
      --bg-secondary: #161b22;
      --bg-tertiary: #21262d;
      --bg-card: #131921;
      --text-primary: #e6edf3;
      --text-secondary: #9198a1;
      --text-muted: #6e7681;
      --border-color: #30363d;
      --accent-primary: #58a6ff;
      --accent-secondary: #3fb950;
      --accent-purple: #d2a8ff;
      --accent-amber: #f0883e;
      --callout-sub-bg: rgba(210, 168, 255, 0.08);
      --callout-sub-border: #a371f7;
      --callout-logic-bg: rgba(88, 166, 255, 0.08);
      --callout-logic-border: #58a6ff;
      --callout-sample-bg: rgba(63, 185, 80, 0.08);
      --callout-sample-border: #3fb950;
      --table-row-alt: rgba(255, 255, 255, 0.025);
      --code-bg: #090d13;
      --shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }}
    :root[data-theme="light"] {{
      --bg-primary: #ffffff;
      --bg-secondary: #f6f8fa;
      --bg-tertiary: #eaeef2;
      --bg-card: #f8fafc;
      --text-primary: #1f2328;
      --text-secondary: #57606a;
      --text-muted: #6e7781;
      --border-color: #d0d7de;
      --accent-primary: #0969da;
      --accent-secondary: #1a7f37;
      --accent-purple: #8250df;
      --accent-amber: #bc4c00;
      --callout-sub-bg: #fbefff;
      --callout-sub-border: #8250df;
      --callout-logic-bg: #ddf4ff;
      --callout-logic-border: #0969da;
      --callout-sample-bg: #dafbe1;
      --callout-sample-border: #1a7f37;
      --table-row-alt: #f6f8fa;
      --code-bg: #1e242c;
      --shadow: 0 6px 18px rgba(31, 35, 40, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    html {{
      scroll-behavior: smooth;
    }}
    body {{
      margin: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.72;
      font-size: 16px;
    }}
    /* Top Navbar */
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      height: 60px;
      background: var(--bg-secondary);
      border-bottom: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 1.25rem;
      backdrop-filter: blur(10px);
    }}
    .topbar-left {{
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }}
    .brand-link {{
      color: var(--text-primary);
      text-decoration: none;
      font-weight: 800;
      font-size: 1.05rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .brand-pill {{
      background: linear-gradient(135deg, #238636, #1f6feb);
      color: #fff;
      font-size: 0.72rem;
      padding: 0.18rem 0.55rem;
      border-radius: 999px;
      font-weight: 700;
      letter-spacing: 0.03em;
    }}
    .topbar-actions {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
    }}
    .btn-action {{
      background: var(--bg-tertiary);
      color: var(--text-primary);
      border: 1px solid var(--border-color);
      padding: 0.4rem 0.8rem;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.15s ease;
    }}
    .btn-action:hover {{
      border-color: var(--accent-primary);
      color: var(--accent-primary);
    }}
    /* 3-Column Layout */
    .layout {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr) 260px;
      max-width: 1720px;
      margin: 0 auto;
    }}
    /* Left Sidebar */
    .sidebar-left {{
      position: sticky;
      top: 60px;
      height: calc(100vh - 60px);
      overflow-y: auto;
      background: var(--bg-secondary);
      border-right: 1px solid var(--border-color);
      padding: 1rem 0.85rem 2rem;
    }}
    .search-box {{
      width: 100%;
      padding: 0.55rem 0.75rem;
      border-radius: 8px;
      border: 1px solid var(--border-color);
      background: var(--bg-primary);
      color: var(--text-primary);
      font-size: 0.86rem;
      margin-bottom: 1rem;
      outline: none;
    }}
    .search-box:focus {{
      border-color: var(--accent-primary);
    }}
    .nav-group {{
      margin-bottom: 1.15rem;
    }}
    .nav-group-title {{
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-secondary);
      padding: 0.35rem 0.5rem;
    }}
    .nav-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.4rem;
      padding: 0.46rem 0.6rem;
      margin: 2px 0;
      border-radius: 7px;
      color: var(--text-primary);
      text-decoration: none;
      font-size: 0.86rem;
      font-weight: 500;
      transition: background 0.15s, color 0.15s;
    }}
    .nav-item:hover {{
      background: var(--bg-tertiary);
      color: var(--accent-primary);
    }}
    .nav-item.active {{
      background: rgba(88, 166, 255, 0.15);
      color: var(--accent-primary);
      font-weight: 700;
      border-left: 3px solid var(--accent-primary);
    }}
    .nav-badge {{
      font-size: 0.68rem;
      padding: 0.1rem 0.42rem;
      border-radius: 99px;
      background: var(--bg-tertiary);
      color: var(--text-secondary);
      white-space: nowrap;
      flex-shrink: 0;
    }}
    /* Main Content Area */
    .main-content {{
      min-width: 0;
      padding: 2.2rem 3rem 4rem;
      max-width: 1020px;
      margin: 0 auto;
      width: 100%;
    }}
    .doc-meta-bar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      padding: 0.65rem 1rem;
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      margin-bottom: 1.5rem;
      font-size: 0.84rem;
      color: var(--text-secondary);
    }}
    .doc-meta-bar code {{
      background: var(--bg-tertiary);
      padding: 0.15rem 0.45rem;
      border-radius: 5px;
      color: var(--accent-primary);
    }}
    /* Typography */
    .markdown-body h1 {{
      font-size: 2.05rem;
      font-weight: 800;
      line-height: 1.25;
      margin-top: 0.5rem;
      margin-bottom: 1.25rem;
      padding-bottom: 0.65rem;
      border-bottom: 2px solid var(--border-color);
    }}
    .markdown-body h2 {{
      font-size: 1.48rem;
      font-weight: 700;
      margin-top: 2.4rem;
      margin-bottom: 0.9rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--border-color);
      color: var(--accent-primary);
    }}
    .markdown-body h3 {{
      font-size: 1.18rem;
      font-weight: 700;
      margin-top: 1.75rem;
      margin-bottom: 0.65rem;
    }}
    .markdown-body h4 {{
      font-size: 1.02rem;
      font-weight: 600;
      margin-top: 1.35rem;
      margin-bottom: 0.45rem;
      color: var(--accent-purple);
    }}
    .markdown-body .headerlink {{
      opacity: 0;
      margin-left: 0.4rem;
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.85em;
      transition: opacity 0.15s;
    }}
    .markdown-body h1:hover .headerlink,
    .markdown-body h2:hover .headerlink,
    .markdown-body h3:hover .headerlink,
    .markdown-body h4:hover .headerlink {{
      opacity: 1;
    }}
    .markdown-body a {{
      color: var(--accent-primary);
      text-decoration: none;
      font-weight: 500;
    }}
    .markdown-body a:hover {{
      text-decoration: underline;
    }}
    .markdown-body hr {{
      border: 0;
      border-top: 1px solid var(--border-color);
      margin: 2rem 0;
    }}
    .markdown-body img {{
      max-width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      margin: 0.75rem 0;
    }}
    /* Code & Pre */
    .markdown-body code {{
      font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.87em;
      background: var(--bg-tertiary);
      padding: 0.16em 0.42em;
      border-radius: 5px;
    }}
    .markdown-body pre {{
      background: var(--code-bg);
      color: #e6edf3;
      padding: 1.1rem 1.25rem;
      border-radius: 10px;
      border: 1px solid var(--border-color);
      overflow-x: auto;
      box-shadow: var(--shadow);
      margin: 1.15rem 0;
    }}
    .markdown-body pre code {{
      background: transparent;
      padding: 0;
      color: inherit;
      font-size: 0.86rem;
      line-height: 1.55;
    }}
    /* Tables */
    .markdown-body table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1.35rem 0;
      font-size: 0.92rem;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      display: block;
      overflow-x: auto;
    }}
    .markdown-body th,
    .markdown-body td {{
      border: 1px solid var(--border-color);
      padding: 0.65rem 0.9rem;
      text-align: left;
      vertical-align: top;
    }}
    .markdown-body th {{
      background: var(--bg-secondary);
      font-weight: 700;
      color: var(--accent-primary);
    }}
    .markdown-body tr:nth-child(even) {{
      background: var(--table-row-alt);
    }}
    /* Math Blocks */
    .math-block {{
      overflow-x: auto;
      padding: 0.65rem 0.9rem;
      margin: 0.9rem 0;
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
    }}
    /* Callout Blockquotes */
    .callout {{
      margin: 1.35rem 0;
      padding: 1rem 1.25rem;
      border-left: 4px solid var(--accent-primary);
      background: var(--bg-secondary);
      border-radius: 0 10px 10px 0;
    }}
    .callout > :first-child {{
      margin-top: 0;
    }}
    .callout > :last-child {{
      margin-bottom: 0;
    }}
    .callout-substitution {{
      background: var(--callout-sub-bg);
      border-left: 4px solid var(--callout-sub-border);
      box-shadow: 0 4px 14px rgba(163, 113, 247, 0.08);
    }}
    .callout-logic {{
      background: var(--callout-logic-bg);
      border-left: 4px solid var(--callout-logic-border);
    }}
    .callout-sample {{
      background: var(--callout-sample-bg);
      border-left: 4px solid var(--callout-sample-border);
    }}
    .callout-warning {{
      background: rgba(240, 136, 62, 0.1);
      border-left: 4px solid var(--accent-amber);
    }}
    /* Right TOC Sidebar */
    .sidebar-right {{
      position: sticky;
      top: 60px;
      height: calc(100vh - 60px);
      overflow-y: auto;
      padding: 1.25rem 1rem 2rem;
      border-left: 1px solid var(--border-color);
      font-size: 0.82rem;
    }}
    .toc-header {{
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-secondary);
      margin-bottom: 0.75rem;
      font-size: 0.75rem;
    }}
    .toc-link {{
      display: block;
      color: var(--text-secondary);
      text-decoration: none;
      padding: 0.28rem 0.45rem;
      border-radius: 5px;
      line-height: 1.4;
      margin-bottom: 2px;
      transition: color 0.15s, background 0.15s;
    }}
    .toc-link:hover {{
      color: var(--accent-primary);
      background: var(--bg-secondary);
    }}
    .toc-l3 {{
      padding-left: 1.15rem;
      font-size: 0.78rem;
      color: var(--text-muted);
    }}
    /* Pager */
    .doc-pager {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border-color);
    }}
    .pager-card {{
      display: flex;
      flex-direction: column;
      padding: 0.9rem 1.1rem;
      border: 1px solid var(--border-color);
      border-radius: 10px;
      background: var(--bg-secondary);
      text-decoration: none;
      color: var(--text-primary);
      transition: border-color 0.15s, transform 0.15s;
    }}
    .pager-card:hover {{
      border-color: var(--accent-primary);
      transform: translateY(-2px);
    }}
    .pager-next {{
      text-align: right;
    }}
    .pager-label {{
      font-size: 0.76rem;
      color: var(--text-secondary);
      font-weight: 600;
      margin-bottom: 0.25rem;
    }}
    .pager-title {{
      font-size: 0.94rem;
      font-weight: 700;
      color: var(--accent-primary);
    }}
    /* Responsive */
    @media (max-width: 1240px) {{
      .layout {{
        grid-template-columns: 280px minmax(0, 1fr);
      }}
      .sidebar-right {{
        display: none;
      }}
    }}
    @media (max-width: 860px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar-left {{
        position: static;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--border-color);
      }}
      .main-content {{
        padding: 1.5rem 1.15rem 3rem;
      }}
      .doc-pager {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-left">
      <a class="brand-link" href="{home_rel_link}">
        <span>🚀 Deep RL & Sim-to-Real Portal</span>
        <span class="brand-pill">LOCAL DOCS</span>
      </a>
    </div>
    <div class="topbar-actions">
      <a class="btn-action" href="#interactive-simulator-lab" title="Mở phòng thí nghiệm mô phỏng tương tác">🎛️ Mô phỏng &amp; Tham số</a>
      <a class="btn-action" href="{md_rel_link}" title="Xem file Markdown gốc">📄 File .md gốc</a>
      <button class="btn-action" id="theme-toggle" onclick="toggleTheme()">🌗 Đổi giao diện Sáng/Tối</button>
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar-left">
      <input
        type="text"
        id="doc-search"
        class="search-box"
        placeholder="🔍 Tìm nhanh bài giảng, chủ đề..."
        oninput="filterSidebar(this.value)"
      />
      <nav id="sidebar-nav">
        {sidebar_html}
      </nav>
    </aside>

    <main class="main-content">
      <div class="doc-meta-bar">
        <div>📚 <strong>Tài liệu:</strong> {page_title}</div>
        <div>📂 Đường dẫn gốc: <code>{md_path_str}</code></div>
      </div>
      {interactive_lab_html}
      <article class="markdown-body">
        {body_html}
      </article>
      {pager_html}
    </main>

    <aside class="sidebar-right">
      <div class="toc-header">📑 Mục lục trên trang này</div>
      <nav class="toc-nav">
        {toc_html}
      </nav>
    </aside>
  </div>

  <script>
    // Khởi tạo giao diện Sáng/Tối từ localStorage
    (function() {{
      const saved = localStorage.getItem('rl_docs_theme') || 'dark';
      document.documentElement.setAttribute('data-theme', saved);
    }})();

    function toggleTheme() {{
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('rl_docs_theme', next);
    }}

    function filterSidebar(query) {{
      const q = query.trim().toLowerCase();
      const items = document.querySelectorAll('#sidebar-nav .nav-item');
      items.forEach(item => {{
        const title = item.getAttribute('data-title') || '';
        item.style.display = (!q || title.includes(q)) ? 'flex' : 'none';
      }});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      if (window.hljs) {{
        hljs.highlightAll();
      }}
    }});
  </script>
</body>
</html>
"""


def build_all():
    generated = []
    for idx, doc in enumerate(DOCS_MANIFEST):
        md_abs = ROOT_DIR / doc["md_path"]
        if not md_abs.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp Markdown: {md_abs}")

        md_text = md_abs.read_text(encoding="utf-8")
        body_html, toc_html = convert_markdown_with_math(md_text, doc["md_path"])

        sidebar_html = build_sidebar_html(doc["id"], doc["html_path"])
        pager_html = build_prev_next_html(idx, doc["html_path"])
        interactive_lab_html = get_interactive_lab_html(doc["id"])

        current_dir = (ROOT_DIR / doc["html_path"]).parent
        home_rel_link = os.path.relpath(ROOT_DIR / "index.html", current_dir).replace("\\", "/")
        md_rel_link = os.path.relpath(md_abs, current_dir).replace("\\", "/")

        full_html = HTML_TEMPLATE.format(
            page_title=html.escape(doc["title"]),
            home_rel_link=home_rel_link,
            md_rel_link=md_rel_link,
            sidebar_html=sidebar_html,
            md_path_str=html.escape(doc["md_path"]),
            interactive_lab_html=interactive_lab_html,
            body_html=body_html,
            pager_html=pager_html,
            toc_html=toc_html,
        )

        html_abs = ROOT_DIR / doc["html_path"]
        html_abs.write_text(full_html, encoding="utf-8")
        generated.append(doc["html_path"])

        if doc.get("extra_html_path"):
            extra_abs = ROOT_DIR / doc["extra_html_path"]
            extra_abs.write_text(full_html, encoding="utf-8")
            generated.append(doc["extra_html_path"])

    print(f"Đã biên dịch thành công {len(generated)} tệp HTML:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    build_all()
