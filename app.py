#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颈椎智能分析平台
基于 YOLO26-KACA ONNX 模型，支持 X 线影像关键点检测、手动拖拽编辑与颈椎指标自动测算
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import json
import base64
import io
import math
import datetime
from pathlib import Path
import pandas as pd

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="颈椎智能分析平台",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  模型自动下载（HuggingFace）
# ══════════════════════════════════════════════════════════════════════════════
MODEL_URL  = "https://huggingface.co/BUCMGYY/cervical-kaca/resolve/main/best.onnx"
MODEL_PATH = Path("best.onnx")


def download_model():
    progress_bar = st.progress(0, text="正在连接 HuggingFace...")
    status_text  = st.empty()

    def reporthook(block_num, block_size, total):
        if total > 0:
            pct = min(block_num * block_size / total, 1.0)
            mb_done  = block_num * block_size / 1024 / 1024
            mb_total = total / 1024 / 1024
            progress_bar.progress(pct,
                text=f"下载中... {mb_done:.1f} / {mb_total:.1f} MB")
    try:
        status_text.info("首次运行，正在从 HuggingFace 下载模型（约 45 MB）...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook)
        progress_bar.progress(1.0, text="✅ 模型下载完成！")
        status_text.empty()
        return True
    except urllib.error.URLError as e:
        progress_bar.empty()
        status_text.error(f"下载失败：{e}\n请检查网络后刷新重试。")
        return False


@st.cache_resource(show_spinner=False)
def ensure_model_cached():
    # best.onnx 已随仓库部署，无需下载
    if not MODEL_PATH.exists():
        return download_model()  # 保留兜底下载逻辑
    return True

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer, Image as RL_Image)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    HAS_PDF = True
except ImportError:
    HAS_PDF = False




# ══════════════════════════════════════════════════════════════════════════════
#  关键点定义（35个关键点，与训练标注一致）
# ══════════════════════════════════════════════════════════════════════════════
# 椎体关键点索引映射（每椎体含 AS/AI/PS/PI/SP，C3-C7另含LP）
# AS=前上, AI=前下, PS=后上, PI=后下, SP=棘突, LP=椎板点
VERTEBRAE = {
    'C2': {'AS': 0,  'AI': 1,  'PS': 2,  'PI': 3,  'SP': 4},
    'C3': {'AS': 5,  'AI': 6,  'PS': 7,  'PI': 8,  'SP': 9,  'LP': 10},
    'C4': {'AS': 11, 'AI': 12, 'PS': 13, 'PI': 14, 'SP': 15, 'LP': 16},
    'C5': {'AS': 17, 'AI': 18, 'PS': 19, 'PI': 20, 'SP': 21, 'LP': 22},
    'C6': {'AS': 23, 'AI': 24, 'PS': 25, 'PI': 26, 'SP': 27, 'LP': 28},
    'C7': {'AS': 29, 'AI': 30, 'PS': 31, 'PI': 32, 'SP': 33, 'LP': 34},
}

# 关键点名称列表（35个，用于显示）
KP_NAMES = [
    # C2 (0-4)
    'C2-前上(AS)', 'C2-前下(AI)', 'C2-后上(PS)', 'C2-后下(PI)', 'C2-棘突(SP)',
    # C3 (5-10)
    'C3-前上(AS)', 'C3-前下(AI)', 'C3-后上(PS)', 'C3-后下(PI)', 'C3-棘突(SP)', 'C3-椎板(LP)',
    # C4 (11-16)
    'C4-前上(AS)', 'C4-前下(AI)', 'C4-后上(PS)', 'C4-后下(PI)', 'C4-棘突(SP)', 'C4-椎板(LP)',
    # C5 (17-22)
    'C5-前上(AS)', 'C5-前下(AI)', 'C5-后上(PS)', 'C5-后下(PI)', 'C5-棘突(SP)', 'C5-椎板(LP)',
    # C6 (23-28)
    'C6-前上(AS)', 'C6-前下(AI)', 'C6-后上(PS)', 'C6-后下(PI)', 'C6-棘突(SP)', 'C6-椎板(LP)',
    # C7 (29-34)
    'C7-前上(AS)', 'C7-前下(AI)', 'C7-后上(PS)', 'C7-后下(PI)', 'C7-棘突(SP)', 'C7-椎板(LP)',
]

# 关键点颜色（按椎体分组）
KP_COLORS = [
    # C2: 蓝
    '#4169E1','#4169E1','#4169E1','#4169E1','#4169E1',
    # C3: 青
    '#00BFFF','#00BFFF','#00BFFF','#00BFFF','#00BFFF','#00BFFF',
    # C4: 绿
    '#32CD32','#32CD32','#32CD32','#32CD32','#32CD32','#32CD32',
    # C5: 金黄
    '#FFD700','#FFD700','#FFD700','#FFD700','#FFD700','#FFD700',
    # C6: 橙
    '#FFA500','#FFA500','#FFA500','#FFA500','#FFA500','#FFA500',
    # C7: 红
    '#FF4500','#FF4500','#FF4500','#FF4500','#FF4500','#FF4500',
]


# ══════════════════════════════════════════════════════════════════════════════
#  7 个静态指标 + 4 个动态指标定义
# ══════════════════════════════════════════════════════════════════════════════
MEAS_DEFS = {
    # ── 静态指标 ──
    'Cobb角 (Cobb Angle)':
        ('C2-PI 与 C7-PI 所在线段夹角，反映颈椎整体前凸/后凸', '°', '正常：20°~40°（前凸为正）'),
    'Ishihara曲率指数':
        ('C3-PI/C4-PI/C5-PI/C6-PI 至 C2-PI—C7-PI 连线垂直距离之和/D×100%', '%', '正常：>12%'),
    'cSVA (颈椎矢状垂直轴)':
        ('C2椎体重心 X 坐标 − C7-PS X 坐标（横向偏移量）', 'px', '临床参考：<18 mm（需比例尺）'),
    '椎体滑移 (Vertebral Slip)':
        ('各相邻椎体 PI-PS 间最大水平错位距离', 'px', '参考：越小越好'),
    '高深比 (H/D Ratio)':
        ('椎体平均高度 / 下终板前后径，反映椎体形态', '比值', '参考值约 0.8~1.2'),
    '椎间盘高度指数 (DHI)':
        ('相邻椎体间前后盘高均值 / 上椎体下终板前后径', '比值', '正常：DHI 越大越好'),
    '椎管直径 (Canal Diameter)':
        ('C3-C7 后中点至椎板点距离均值，反映椎管矢状径', 'px', '成人参考：>13 mm（需比例尺）'),
    # ── 动态指标（需屈伸位双图） ──
    '节段活动度 (Segmental ROM)':
        ('屈伸位各节段 Cobb 角变化均值（需配对屈伸位图像）', '°', ''),
    '颈椎总活动度 (Total ROM)':
        ('屈伸位整体 Cobb 角差值（需配对屈伸位图像）', '°', '正常：>60°'),
    '节段平移 (Segmental Translation)':
        ('屈伸位各节段水平位移变化最大值（需配对屈伸位图像）', 'px', '临床参考：<3mm（需比例尺）'),
    '棘突间距变化 (ISD)':
        ('屈伸位棘突间距变化均值（需配对屈伸位图像）', 'px', ''),
}

# 静态指标列表（单图可计算）
STATIC_MEAS = [k for k in list(MEAS_DEFS.keys())[:7]]
# 动态指标列表（需屈伸双图）
DYNAMIC_MEAS = [k for k in list(MEAS_DEFS.keys())[7:]]


# ══════════════════════════════════════════════════════════════════════════════
#  几何工具函数（与标准代码一致）
# ══════════════════════════════════════════════════════════════════════════════
def pt(kps, i):
    """取第 i 个关键点的 (x, y) numpy 向量"""
    return np.array([float(kps[i][0]), float(kps[i][1])])

def mid(a, b):
    return (a + b) / 2.0

def edist(a, b):
    return np.linalg.norm(a - b)

def angle2v(v1, v2):
    """两向量夹角（度）"""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
    c = np.dot(v1, v2) / denom
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

def perp_dist(point, lp1, lp2):
    """点到直线的垂直距离"""
    lv = lp2 - lp1
    ll = np.linalg.norm(lv)
    if ll < 1e-12:
        return float(np.linalg.norm(point - lp1))
    lu = lv / ll
    pp = lp1 + np.dot(point - lp1, lu) * lu
    return float(np.linalg.norm(point - pp))


# ══════════════════════════════════════════════════════════════════════════════
#  7 个静态指标计算
# ══════════════════════════════════════════════════════════════════════════════
def calc_cobb_angle(k):
    """C2-PI(idx=3) 与 C7-PI(idx=32) 所在终板线的夹角"""
    # C2 下终板: PI(3)→AI(1) 方向向量
    # C7 下终板: PI(32)→AI(30) 方向向量
    return round(angle2v(pt(k, 3) - pt(k, 1),
                         pt(k, 32) - pt(k, 30)), 2)

def calc_ishihara_index(k):
    """Ishihara 颈椎曲率指数
    参考线: C2-PI(3) 到 C7-PI(32)
    测量点: C3-PI(8), C4-PI(14), C5-PI(20), C6-PI(26)
    公式: (a3+a4+a5+a6) / D × 100%
    """
    p2, p7 = pt(k, 3), pt(k, 32)
    D = edist(p2, p7)
    if D < 1e-6:
        return 0.0
    val = sum(perp_dist(pt(k, i), p2, p7) for i in [8, 14, 20, 26])
    return round(val / D * 100, 2)

def calc_csva(k):
    """cSVA: C2椎体重心(4点均值) X 坐标 - C7-PS(31) X 坐标
    正值=重心前移（不稳定），负值=后移
    """
    c2_cen = (pt(k, 0) + pt(k, 1) + pt(k, 2) + pt(k, 3)) / 4.0
    return round(float(c2_cen[0] - pt(k, 31)[0]), 2)

def calc_hd_ratio(k):
    """H/D Ratio（高深比）
    对 C3~C7 每个椎体: H_avg = (Ha + Hm + Hp) / 3，VBd = dist(AI, PI)
    Ha = dist(AS, AI), Hm = dist(mid(AS,PS), mid(AI,PI)), Hp = dist(PS, PI)
    返回各椎体均值
    """
    rs = []
    for v in ['C3', 'C4', 'C5', 'C6', 'C7']:
        d = VERTEBRAE[v]
        Ha  = edist(pt(k, d['AS']), pt(k, d['AI']))
        Hm  = edist(mid(pt(k, d['AS']), pt(k, d['PS'])),
                    mid(pt(k, d['AI']), pt(k, d['PI'])))
        Hp  = edist(pt(k, d['PS']), pt(k, d['PI']))
        VBd = edist(pt(k, d['AI']), pt(k, d['PI']))
        if VBd > 1e-6:
            rs.append((Ha + Hm + Hp) / 3.0 / VBd)
    return round(float(np.mean(rs)), 4) if rs else 0.0

def calc_slip(k):
    """椎体滑移: 各相邻椎 PI-PS 对间最大水平错位
    配对: C2-PI(3)↔C3-PS(7), C3-PI(8)↔C4-PS(13),
           C4-PI(14)↔C5-PS(19), C5-PI(20)↔C6-PS(25), C6-PI(26)↔C7-PS(31)
    """
    segs = [(3, 7), (8, 13), (14, 19), (20, 25), (26, 31)]
    return round(float(max(abs(pt(k, a)[0] - pt(k, b)[0]) for a, b in segs)), 2)

def calc_dhi(k):
    """椎间盘高度指数 (DHI)
    对 C2-C3 ~ C6-C7 各节段：
    Da = dist(上椎AI, 下椎AS), Dp = dist(上椎PI, 下椎PS)
    VBd = dist(上椎AI, 上椎PI)（上椎下终板前后径）
    DHI = (Da + Dp) / 2 / VBd
    """
    segs = [('C2', 'C3'), ('C3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'C7')]
    ds = []
    for u, l in segs:
        U, L = VERTEBRAE[u], VERTEBRAE[l]
        Da  = edist(pt(k, U['AI']), pt(k, L['AS']))
        Dp  = edist(pt(k, U['PI']), pt(k, L['PS']))
        VBd = edist(pt(k, U['AI']), pt(k, U['PI']))
        if VBd > 1e-6:
            ds.append((Da + Dp) / 2.0 / VBd)
    return round(float(np.mean(ds)), 4) if ds else 0.0

def calc_canal(k):
    """椎管直径: C3~C7 各椎体后中点至椎板点(LP)距离均值"""
    ds = []
    for v in ['C3', 'C4', 'C5', 'C6', 'C7']:
        d = VERTEBRAE[v]
        if 'LP' not in d:
            continue
        back_mid = mid(pt(k, d['PS']), pt(k, d['PI']))
        ds.append(edist(back_mid, pt(k, d['LP'])))
    return round(float(np.mean(ds)), 2) if ds else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  4 个动态指标计算（需屈伸位双图）
# ══════════════════════════════════════════════════════════════════════════════
def calc_seg_rom(kf, ke):
    """节段活动度: 屈伸位各节段终板线夹角差均值
    节段配对索引(AI, PI, 下一椎AI, 下一椎PI):
    C2-C3:(1,3,6,8), C3-C4:(6,8,12,14), C4-C5:(12,14,18,20),
    C5-C6:(18,20,24,26), C6-C7:(24,26,30,32)
    """
    segs = [(1,3,6,8), (6,8,12,14), (12,14,18,20), (18,20,24,26), (24,26,30,32)]
    vals = []
    for a, b, c, d in segs:
        af = angle2v(pt(kf,b)-pt(kf,a), pt(kf,d)-pt(kf,c))
        ae = angle2v(pt(ke,b)-pt(ke,a), pt(ke,d)-pt(ke,c))
        vals.append(abs(af - ae))
    return round(float(np.mean(vals)), 2)

def calc_global_rom(kf, ke):
    """颈椎总活动度: 屈伸位整体 Cobb 角之差"""
    return round(abs(calc_cobb_angle(kf) - calc_cobb_angle(ke)), 2)

def calc_seg_trans(kf, ke):
    """节段平移: |[x(Cn-PI)-x(Cn+1-PS)]_屈 - [x(Cn-PI)-x(Cn+1-PS)]_伸| 最大值
    配对: C2-PI(3)↔C3-PS(7), C3-PI(8)↔C4-PS(13),
           C4-PI(14)↔C5-PS(19), C5-PI(20)↔C6-PS(25), C6-PI(26)↔C7-PS(31)
    """
    segs = [(3,7), (8,13), (14,19), (20,25), (26,31)]
    trans = []
    for pi_idx, ps_idx in segs:
        df = pt(kf, pi_idx)[0] - pt(kf, ps_idx)[0]
        de = pt(ke, pi_idx)[0] - pt(ke, ps_idx)[0]
        trans.append(abs(df - de))
    return round(float(max(trans)), 2) if trans else 0.0

def calc_isd_change(kf, ke):
    """棘突间距变化: 屈伸位各相邻棘突间距变化均值
    棘突索引: C2-SP(4), C3-SP(9), C4-SP(15), C5-SP(21), C6-SP(27), C7-SP(33)
    """
    sp = [4, 9, 15, 21, 27, 33]
    diffs = []
    for i in range(len(sp) - 1):
        df = edist(pt(kf, sp[i]), pt(kf, sp[i+1]))
        de = edist(pt(ke, sp[i]), pt(ke, sp[i+1]))
        diffs.append(df - de)
    return round(float(np.mean(diffs)), 2) if diffs else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  统一指标计算入口
# ══════════════════════════════════════════════════════════════════════════════
def compute_measurements(kp: np.ndarray,
                         kp_ext: np.ndarray = None,
                         selected: list = None) -> dict:
    """
    kp:      (35, 2) 侧位/屈曲位关键点坐标（像素）
    kp_ext:  (35, 2) 伸展位关键点坐标（可选，用于动态指标）
    selected: 指标名称列表（None=全部静态）
    返回: {指标名: 数值字符串}
    """
    if selected is None:
        selected = STATIC_MEAS

    res = {}
    for name in selected:
        try:
            if name == 'Cobb角 (Cobb Angle)':
                res[name] = f"{calc_cobb_angle(kp)}°"
            elif name == 'Ishihara曲率指数':
                res[name] = f"{calc_ishihara_index(kp)}%"
            elif name == 'cSVA (颈椎矢状垂直轴)':
                res[name] = f"{calc_csva(kp)} px"
            elif name == '椎体滑移 (Vertebral Slip)':
                res[name] = f"{calc_slip(kp)} px"
            elif name == '高深比 (H/D Ratio)':
                res[name] = f"{calc_hd_ratio(kp)}"
            elif name == '椎间盘高度指数 (DHI)':
                res[name] = f"{calc_dhi(kp)}"
            elif name == '椎管直径 (Canal Diameter)':
                res[name] = f"{calc_canal(kp)} px"
            # ── 动态指标（需屈伸双图）──
            elif name in DYNAMIC_MEAS:
                if kp_ext is None:
                    res[name] = '需上传伸展位图像'
                elif name == '节段活动度 (Segmental ROM)':
                    res[name] = f"{calc_seg_rom(kp, kp_ext)}°"
                elif name == '颈椎总活动度 (Total ROM)':
                    res[name] = f"{calc_global_rom(kp, kp_ext)}°"
                elif name == '节段平移 (Segmental Translation)':
                    res[name] = f"{calc_seg_trans(kp, kp_ext)} px"
                elif name == '棘突间距变化 (ISD)':
                    res[name] = f"{calc_isd_change(kp, kp_ext)} px"
            else:
                res[name] = 'N/A'
        except Exception as e:
            res[name] = f'计算错误: {e}'
    return res

# ══════════════════════════════════════════════════════════════════════════════
#  ONNX 推理
# ══════════════════════════════════════════════════════════════════════════════
def letterbox(img: np.ndarray, new_shape=640, color=(114,114,114)):
    """保持比例缩放+填充至正方形"""
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    nw, nh = int(w * r), int(h * r)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top    = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left   = (new_shape - nw) // 2
    right  = new_shape - nw - left
    img_p = cv2.copyMakeBorder(img_r, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=color)
    return img_p, r, (left, top)


def preprocess_image(pil_img: Image.Image, imgsz=640):
    img_np  = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_lb, ratio, pad = letterbox(img_bgr, imgsz)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_f   = img_rgb.astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img_f, (2,0,1)), 0), ratio, pad


def postprocess_output(raw, ratio, pad, conf_thr=0.25, iou_thr=0.45):
    """
    兼容两种输出格式:
      (1) (1, 300, 111)  — Ultralytics 含NMS后处理
      (2) (1, 111, 8400) — 原始锚点输出，需手动NMS
    """
    arr = raw[0]  # 去掉 batch 维

    # 自动判断格式
    if arr.shape[0] < arr.shape[1]:
        # (111, 8400) → (8400, 111)
        arr = arr.T

    # arr: (N, 111) — [x1,y1,x2,y2, conf, cls, kp0_x,kp0_y,kp0_c, ...]
    # 或 (N, 111)  — [cx,cy,w,h, conf, cls, ...]
    conf_col = arr[:, 4]
    mask = conf_col > conf_thr
    arr  = arr[mask]
    if len(arr) == 0:
        return []

    # 判断是 cx/cy/w/h 还是 x1/y1/x2/y2
    # 若 x1 < 1 且 x2 > x1 成立则已是 xyxy；否则是 cxcywh
    boxes = arr[:, :4].copy()
    if np.median(boxes[:, 2]) > np.median(boxes[:, 0]):
        # 已是 xyxy（或 xyxy scaled to img size）
        boxes_xyxy = boxes
    else:
        # cxcywh → xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    scores = arr[:, 4].tolist()
    try:
        indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores, conf_thr, iou_thr)
        indices = indices.flatten() if len(indices) > 0 else []
    except Exception:
        indices = [np.argmax(scores)]

    results = []
    for idx in indices:
        row  = arr[idx]
        conf = float(row[4])
        cls  = int(row[5]) if row.shape[0] > 6 else 0
        kpts = row[6:111].reshape(35, 3)          # (35, x/y/kconf)

        # 还原 letterbox 变换
        kp_xy = kpts[:, :2].copy()
        kp_xy[:, 0] = (kp_xy[:, 0] - pad[0]) / ratio
        kp_xy[:, 1] = (kp_xy[:, 1] - pad[1]) / ratio

        results.append({
            "conf":    conf,
            "kp_xy":   kp_xy,           # (35, 2) 原图坐标
            "kp_conf": kpts[:, 2],       # (35,)
        })

    results.sort(key=lambda x: x["conf"], reverse=True)
    return results


@st.cache_resource(show_spinner="加载模型中...")
def load_onnx_session(model_path: str):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, opts,
        providers=["CPUExecutionProvider"]
    )


def run_inference(session, pil_img: Image.Image, conf_thr=0.25):
    inp, ratio, pad = preprocess_image(pil_img)
    in_name = session.get_inputs()[0].name
    out     = session.run(None, {in_name: inp})
    return postprocess_output(out[0], ratio, pad, conf_thr)


# ══════════════════════════════════════════════════════════════════════════════
#  图像绘制

def pil_to_data_url(pil_img: Image.Image) -> str:
    """PIL Image → base64 data URL，用于 st_canvas background_image_url"""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# ══════════════════════════════════════════════════════════════════════════════
def draw_kp_on_image(pil_img: Image.Image, kp_xy: np.ndarray,
                     kp_conf: np.ndarray = None, radius=6) -> Image.Image:
    img_out = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_out.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    for i, (x, y) in enumerate(kp_xy):
        c = float(kp_conf[i]) if kp_conf is not None else 1.0
        if c < 0.05:
            continue
        x, y = int(round(x)), int(round(y))
        hex_c = KP_COLORS[i].lstrip("#")
        r2, g2, b2 = int(hex_c[:2],16), int(hex_c[2:4],16), int(hex_c[4:],16)
        # 填充圆（半透明）
        draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                     fill=(r2,g2,b2,140), outline=(r2,g2,b2,255), width=2)
        # 编号标签
        draw.text((x+radius+2, y-radius), str(i), fill=(r2,g2,b2,220))

    img_out = Image.alpha_composite(img_out, overlay).convert("RGB")
    return img_out


def kp_to_fabric_json(kp_xy, kp_conf, dw, dh, ow, oh):
    """将关键点坐标转换为 Fabric.js JSON，用于 drawable-canvas 初始化"""
    sx, sy = dw / ow, dh / oh
    objects = []
    for i, (x, y) in enumerate(kp_xy):
        c = float(kp_conf[i]) if kp_conf is not None else 1.0
        color = KP_COLORS[i]
        objects.append({
            "type": "circle",
            "version": "4.4.0",
            "originX": "center", "originY": "center",
            "left": float(x) * sx,
            "top":  float(y) * sy,
            "radius": 7,
            "fill": color + "88",
            "stroke": color,
            "strokeWidth": 2,
            "selectable": True,
            "hasControls": False,
            "hasBorders": False,
            "lockScalingX": True,
            "lockScalingY": True,
            "opacity": 1.0 if c > 0.05 else 0.3,
            "name": KP_NAMES[i],
        })
    return {"version": "4.4.0", "objects": objects}


def fabric_to_kp(json_data, dw, dh, ow, oh) -> np.ndarray:
    """从 Fabric.js canvas 结果提取关键点坐标"""
    sx, sy = ow / dw, oh / dh
    objects = json_data.get("objects", [])
    kp = np.zeros((35, 2))
    for i, obj in enumerate(objects[:35]):
        if obj.get("type") == "circle":
            kp[i, 0] = obj.get("left", 0) * sx
            kp[i, 1] = obj.get("top",  0) * sy
    return kp


# ══════════════════════════════════════════════════════════════════════════════
#  PDF 报告生成
# ══════════════════════════════════════════════════════════════════════════════
def make_pdf_report(patient_info, annotated_img, selected, meas_res):
    if not HAS_PDF:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm,   bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story  = []

    # ── 标题 ──
    title_sty = ParagraphStyle("t", parent=styles["Title"],
                                fontSize=20, alignment=1, spaceAfter=6)
    story.append(Paragraph("颈椎影像智能分析报告", title_sty))
    sub_sty = ParagraphStyle("s", parent=styles["Normal"],
                              fontSize=9, textColor=rl_colors.grey, alignment=1)
    story.append(Paragraph(
        f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}　　"
        f"系统：YOLO26-KACA 颈椎关键点识别平台", sub_sty))
    story.append(Spacer(1, 0.4*cm))

    # ── 患者信息 ──
    info_data = [
        ["姓　　名", patient_info.get("name","—"),
         "检查日期", datetime.date.today().strftime("%Y-%m-%d")],
        ["年　　龄", patient_info.get("age","—"),
         "性　　别", patient_info.get("gender","—")],
        ["病　案　号", patient_info.get("case_id","—"),
         "科　　室", patient_info.get("dept","脊柱科")],
    ]
    info_tbl = Table(info_data, colWidths=[3*cm,4.5*cm,3*cm,4.5*cm])
    info_tbl.setStyle(TableStyle([
        ("GRID",      (0,0),(-1,-1), 0.4, rl_colors.grey),
        ("FONTSIZE",  (0,0),(-1,-1), 10),
        ("BACKGROUND",(0,0),(0,-1),  rl_colors.HexColor("#E8EDF2")),
        ("BACKGROUND",(2,0),(2,-1),  rl_colors.HexColor("#E8EDF2")),
        ("FONTNAME",  (0,0),(-1,-1), "Helvetica"),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── 标注图像 ──
    story.append(Paragraph("影像标注结果", styles["Heading2"]))
    img_buf = io.BytesIO()
    annotated_img.save(img_buf, format="PNG")
    img_buf.seek(0)
    iw, ih = annotated_img.size
    disp_w = 12 * cm
    disp_h = disp_w * ih / iw
    story.append(RL_Image(img_buf, width=disp_w, height=disp_h))
    story.append(Spacer(1, 0.5*cm))

    # ── 测量结果表 ──
    story.append(Paragraph("颈椎指标测量结果", styles["Heading2"]))
    normal_map = {
        "C2-C7 Cobb角":       "20°~40°（前凸为正）",
        "T1 倾斜角":          "<25°",
        "C2-C7 SVA":          "<18 mm（需比例尺校正）",
        "颈椎曲率指数 (CCI)": ">12%",
        "寰齿前间距 (ADI)":   "成人 <3 mm（需比例尺校正）",
    }
    meas_data = [["测量指标", "测量值", "参考正常范围"]]
    for m in selected:
        val = meas_res.get(m, "N/A")
        ref = normal_map.get(m, "—")
        meas_data.append([m, val, ref])

    meas_tbl = Table(meas_data, colWidths=[6*cm, 3.5*cm, 5.5*cm])
    meas_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  rl_colors.HexColor("#2E4057")),
        ("TEXTCOLOR",     (0,0),(-1,0),  rl_colors.white),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("FONTSIZE",      (0,0),(-1,-1), 10),
        ("GRID",          (0,0),(-1,-1), 0.5, rl_colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [rl_colors.white,
                                          rl_colors.HexColor("#F5F7FA")]),
    ]))
    story.append(meas_tbl)
    story.append(Spacer(1, 0.8*cm))

    # ── 免责声明 ──
    disc_sty = ParagraphStyle("disc", parent=styles["Normal"],
                               fontSize=8, textColor=rl_colors.HexColor("#888888"))
    story.append(Paragraph(
        "【免责声明】本报告由人工智能辅助生成，仅供临床参考，不作为最终诊断依据。"
        "所有测量数据以像素为单位（SVA、ADI 等绝对值需结合比例尺换算），"
        "请由具有资质的临床医师结合患者实际情况综合判断。",
        disc_sty
    ))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  CSS 样式
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    /* 整体背景 */
    .stApp { background: #0D1117; color: #E6EDF3; }
    section[data-testid="stSidebar"] { background: #161B22; }

    /* 卡片样式 */
    .metric-card {
        background: #1C2128; border: 1px solid #30363D;
        border-radius: 10px; padding: 16px 20px; margin: 6px 0;
        text-align: center;
    }
    .metric-card .label { font-size: 13px; color: #8B949E; margin-bottom: 4px; }
    .metric-card .value { font-size: 26px; font-weight: 700; color: #58A6FF; }
    .metric-card .ref   { font-size: 11px; color: #6E7681; margin-top: 4px; }

    /* 区块标题 */
    .section-title {
        font-size: 16px; font-weight: 600; color: #58A6FF;
        border-left: 3px solid #58A6FF; padding-left: 10px; margin: 14px 0 8px;
    }

    /* 标签页 */
    .stTabs [role="tab"] { color: #8B949E; }
    .stTabs [role="tab"][aria-selected="true"] { color: #58A6FF; }

    /* 按钮 */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #1F6FEB, #388BFD);
        border: none; color: white; font-weight: 600;
        border-radius: 8px; padding: 10px 20px;
    }
    .stButton>button { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  主应用
# ══════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # ── 侧边栏 ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 🦴 颈椎智能分析平台")
        st.caption("YOLO26-KACA · 35关键点 · ONNX Runtime")
        st.divider()

        # 模型设置
        st.markdown('<div class="section-title">⚙️ 模型设置</div>', unsafe_allow_html=True)
        model_path = st.text_input(
            "ONNX 模型路径",
            value="best.onnx",
            help="将 best.onnx 放置在 app.py 同目录下，或填写绝对路径"
        )
        conf_thr = st.slider("检测置信度阈值", 0.05, 0.95, 0.25, 0.05)
        show_labels = st.checkbox("显示关键点标签", value=True)
        kp_radius   = st.slider("关键点半径", 3, 15, 6)

        st.divider()

        # 患者信息
        st.markdown('<div class="section-title">👤 患者信息</div>', unsafe_allow_html=True)
        p_name   = st.text_input("姓名", key="p_name")
        c1, c2   = st.columns(2)
        p_age    = c1.text_input("年龄", key="p_age")
        p_gender = c2.selectbox("性别", ["男","女","—"], key="p_gender")
        p_case   = st.text_input("病案号", key="p_case")
        p_dept   = st.text_input("科室", "脊柱科", key="p_dept")

        st.divider()

        # 指标选择
        st.markdown('<div class="section-title">📐 测量指标选择</div>', unsafe_allow_html=True)
        all_meas = list(MEAS_DEFS.keys())
        selected_meas = st.multiselect(
            "选择测量项目（可多选）",
            all_meas,
            default=["Cobb角 (Cobb Angle)", "Ishihara曲率指数",
                     "cSVA (颈椎矢状垂直轴)", "椎间盘高度指数 (DHI)",
                     "椎管直径 (Canal Diameter)"],
        )
        # 提示动态指标需要屈伸双图
        if any(m in DYNAMIC_MEAS for m in selected_meas):
            st.info("💡 动态指标需在「图像分析」页上传**屈曲位 + 伸展位**两张图像")
        with st.expander("📋 指标说明"):
            for m in selected_meas:
                desc, unit, normal = MEAS_DEFS[m]
                st.caption(f"**{m}**（{unit}）：{desc}")
                if normal:
                    st.caption(f"　　→ {normal}")

    # ── 主内容区 ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📷 图像分析 & 关键点编辑",
        "📊 指标测量结果",
        "📄 结构化报告导出",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # Tab 1: 图像分析
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        col_upload, col_canvas = st.columns([1, 2])

        with col_upload:
            st.markdown('<div class="section-title">📁 上传影像</div>', unsafe_allow_html=True)
            st.caption("**屈曲位（必选）** / 静态侧位")
            uploaded = st.file_uploader(
                "上传屈曲位或侧位 X 线片",
                type=["png","jpg","jpeg","bmp","tif","tiff"],
                key="upload_flex",
            )
            st.caption("**伸展位（可选）** — 用于动态指标计算")
            uploaded_ext = st.file_uploader(
                "上传伸展位 X 线片（可选）",
                type=["png","jpg","jpeg","bmp","tif","tiff"],
                key="upload_ext",
            )
            if uploaded_ext:
                pil_ext = Image.open(uploaded_ext).convert("RGB")
                st.session_state["pil_img_ext"] = pil_ext
                st.image(pil_ext,
                         caption=f"伸展位 {pil_ext.width}×{pil_ext.height}px",
                         use_container_width=True)

            if uploaded:
                pil_img = Image.open(uploaded).convert("RGB")
                st.session_state["pil_img"] = pil_img
                st.image(pil_img, caption=f"原图 {pil_img.width}×{pil_img.height}px",
                         use_container_width=True)

                st.markdown('<div class="section-title">🚀 AI 推理</div>', unsafe_allow_html=True)

                if not HAS_ORT:
                    st.error("未找到 onnxruntime，请安装：`pip install onnxruntime`")
                elif not Path(model_path).exists():
                    st.warning(f"模型文件不存在：`{model_path}`\n\n请将 `best.onnx` 放到 app.py 同目录。")
                else:
                    if st.button("▶ 运行检测", type="primary", use_container_width=True):
                        with st.spinner("ONNX 推理中..."):
                            try:
                                session = load_onnx_session(model_path)
                                dets    = run_inference(session, pil_img, conf_thr)
                            except Exception as e:
                                st.error(f"推理失败：{e}")
                                dets = []

                        if not dets:
                            st.warning("未检测到目标，请尝试降低置信度阈值")
                        else:
                            det = dets[0]
                            st.session_state["kp_xy"]   = det["kp_xy"].copy()
                            st.session_state["kp_conf"] = det["kp_conf"].copy()
                            st.session_state["canvas_key"] = str(datetime.datetime.now().timestamp())
                            st.success(f"✅ 检测成功，置信度 {det['conf']:.3f}")
                            st.rerun()

        with col_canvas:
            st.markdown('<div class="section-title">🖱️ 关键点编辑</div>', unsafe_allow_html=True)

            if "kp_xy" not in st.session_state:
                st.info("请先上传图像并运行 AI 检测")
            else:
                pil_img  = st.session_state["pil_img"]
                kp_xy    = st.session_state["kp_xy"]
                kp_conf  = st.session_state["kp_conf"]
                ow, oh   = pil_img.size

                # 显示尺寸
                display_w = min(750, ow)
                display_h = int(display_w * oh / ow)

                if HAS_CANVAS:
                    st.caption("💡 选择 **transform** 模式后点选圆点可拖动调整；编辑完成点击「✅ 应用编辑」")

                    init_json = kp_to_fabric_json(
                        kp_xy, kp_conf, display_w, display_h, ow, oh
                    )
                    # 将图像转为 base64 URL（兼容 Streamlit 1.55+）
                    bg_url = pil_to_data_url(
                        pil_img.resize((display_w, display_h))
                    )
                    canvas_result = st_canvas(
                        fill_color="rgba(0,0,0,0)",
                        stroke_width=2,
                        background_image_url=bg_url,
                        update_streamlit=True,
                        height=display_h,
                        width=display_w,
                        drawing_mode="transform",
                        initial_drawing=init_json,
                        key=st.session_state.get("canvas_key","canvas_init"),
                        display_toolbar=True,
                        point_display_radius=7,
                    )

                    if canvas_result.json_data and canvas_result.json_data.get("objects"):
                        new_kp = fabric_to_kp(
                            canvas_result.json_data, display_w, display_h, ow, oh
                        )
                        st.session_state["kp_xy_draft"] = new_kp

                    bcol1, bcol2 = st.columns(2)
                    if bcol1.button("✅ 应用拖动编辑", use_container_width=True):
                        if "kp_xy_draft" in st.session_state:
                            st.session_state["kp_xy"] = st.session_state["kp_xy_draft"].copy()
                            st.success("关键点已更新")
                            st.rerun()
                    if bcol2.button("↩️ 重置为检测结果", use_container_width=True):
                        st.session_state.pop("kp_xy_draft", None)
                        st.session_state["canvas_key"] = str(datetime.datetime.now().timestamp())
                        st.rerun()

                else:
                    # 降级显示：静态标注图
                    st.warning("安装 `streamlit-drawable-canvas` 可启用交互拖拽编辑")
                    annotated = draw_kp_on_image(pil_img, kp_xy, kp_conf, kp_radius)
                    st.image(annotated, use_container_width=True)

                # ── 精细坐标编辑表格 ──
                with st.expander("🔢 关键点坐标精细编辑（数值调整）", expanded=False):
                    df_kp = pd.DataFrame({
                        "关键点": KP_NAMES,
                        "X (px)": np.round(kp_xy[:,0], 1),
                        "Y (px)": np.round(kp_xy[:,1], 1),
                        "置信度": np.round(kp_conf, 3),
                    })
                    edited = st.data_editor(
                        df_kp,
                        use_container_width=True,
                        num_rows="fixed",
                        column_config={
                            "关键点": st.column_config.TextColumn(disabled=True),
                            "置信度": st.column_config.NumberColumn(disabled=True, format="%.3f"),
                        },
                        key="kp_table"
                    )
                    if st.button("📥 应用表格修改", use_container_width=True):
                        new_kp = np.column_stack([
                            edited["X (px)"].values.astype(float),
                            edited["Y (px)"].values.astype(float),
                        ])
                        st.session_state["kp_xy"] = new_kp
                        st.success("坐标已更新")
                        st.rerun()

                # 生成并缓存标注图
                annotated_img = draw_kp_on_image(pil_img, st.session_state["kp_xy"],
                                                 kp_conf, kp_radius)
                st.session_state["annotated_img"] = annotated_img

    # ────────────────────────────────────────────────────────────────────────
    # Tab 2: 指标测量
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        if "kp_xy" not in st.session_state:
            st.info("🔙 请先在「图像分析」标签页完成关键点检测")
        elif not selected_meas:
            st.info("🔙 请在左侧边栏选择要计算的测量指标")
        else:
            kp_xy = st.session_state["kp_xy"]
            kp_ext = st.session_state.get("kp_xy_ext", None)
            all_res = compute_measurements(kp_xy, kp_ext=kp_ext, selected=selected_meas)
            st.session_state["meas_results"] = all_res
            st.session_state["selected_meas"] = selected_meas

            st.markdown('<div class="section-title">📐 测量结果概览</div>',
                        unsafe_allow_html=True)

            # 3列显示 metric card
            cols = st.columns(3)
            for i, m in enumerate(selected_meas):
                val  = all_res.get(m, "N/A")
                desc, unit, normal = MEAS_DEFS[m]
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">{m}</div>
                        <div class="value">{val}</div>
                        <div class="ref">{normal if normal else desc[:30]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.divider()

            # 详细表格
            st.markdown('<div class="section-title">📋 详细测量数据</div>',
                        unsafe_allow_html=True)
            rows = []
            for m in selected_meas:
                desc, unit, normal = MEAS_DEFS[m]
                rows.append({
                    "测量指标": m,
                    "测量值":   all_res.get(m, "N/A"),
                    "参考正常范围": normal if normal else "—",
                    "说明":     desc,
                })
            df_meas = pd.DataFrame(rows)
            st.dataframe(df_meas, use_container_width=True, hide_index=True,
                         column_config={"说明": st.column_config.TextColumn(width="large")})

            # 侧边展示标注图
            if "annotated_img" in st.session_state:
                st.divider()
                st.markdown('<div class="section-title">🖼️ 关键点标注预览</div>',
                            unsafe_allow_html=True)
                st.image(st.session_state["annotated_img"],
                         use_container_width=True, caption="当前关键点分布")

    # ────────────────────────────────────────────────────────────────────────
    # Tab 3: 报告导出
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        if "meas_results" not in st.session_state:
            st.info("🔙 请先完成图像分析和指标测量")
        else:
            meas_res = st.session_state["meas_results"]
            sel      = st.session_state.get("selected_meas", [])
            ann_img  = st.session_state.get("annotated_img")
            kp_xy    = st.session_state["kp_xy"]

            preview_col, export_col = st.columns([1.4, 1])

            with preview_col:
                st.markdown('<div class="section-title">📋 报告预览</div>',
                            unsafe_allow_html=True)

                # 患者信息卡
                st.markdown(f"""
                <div style="background:#1C2128;border:1px solid #30363D;border-radius:10px;padding:14px 18px;margin-bottom:12px;">
                    <b>姓名：</b>{p_name or "—"}　
                    <b>年龄：</b>{p_age or "—"}　
                    <b>性别：</b>{p_gender}　
                    <b>病案号：</b>{p_case or "—"}　
                    <b>科室：</b>{p_dept}
                </div>
                """, unsafe_allow_html=True)

                if ann_img:
                    st.image(ann_img, use_container_width=True)

                rows = [{"指标": m, "值": meas_res.get(m,"N/A")} for m in sel]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with export_col:
                st.markdown('<div class="section-title">⬇️ 导出选项</div>',
                            unsafe_allow_html=True)

                patient_info = {
                    "name": p_name, "age": p_age,
                    "gender": p_gender, "case_id": p_case, "dept": p_dept,
                }
                date_str = datetime.date.today().strftime("%Y%m%d")

                # JSON
                report_json = {
                    "patient":      patient_info,
                    "date":         datetime.datetime.now().isoformat(),
                    "measurements": {m: meas_res.get(m,"N/A") for m in sel},
                    "keypoints":    {
                        KP_NAMES[i]: {"x": float(kp_xy[i,0]), "y": float(kp_xy[i,1])}
                        for i in range(35)
                    },
                }
                st.download_button(
                    "⬇️ 下载 JSON 数据",
                    data=json.dumps(report_json, ensure_ascii=False, indent=2),
                    file_name=f"cervical_report_{date_str}.json",
                    mime="application/json",
                    use_container_width=True,
                )

                # CSV
                csv_df = pd.DataFrame([
                    {"指标": m, "测量值": meas_res.get(m,"N/A"),
                     "说明": MEAS_DEFS[m][0], "参考正常值": MEAS_DEFS[m][2]}
                    for m in sel
                ])
                st.download_button(
                    "⬇️ 下载 CSV 数据",
                    data=csv_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name=f"cervical_measurements_{date_str}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                # 标注图
                if ann_img:
                    img_buf = io.BytesIO()
                    ann_img.save(img_buf, format="PNG")
                    st.download_button(
                        "⬇️ 下载标注影像 (PNG)",
                        data=img_buf.getvalue(),
                        file_name=f"annotated_{date_str}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                # PDF
                st.divider()
                if HAS_PDF:
                    if st.button("📄 生成 PDF 报告", use_container_width=True, type="primary"):
                        with st.spinner("生成 PDF 中..."):
                            pdf_bytes = make_pdf_report(
                                patient_info, ann_img or Image.new("RGB",(100,100)),
                                sel, meas_res
                            )
                        if pdf_bytes:
                            st.download_button(
                                "⬇️ 下载 PDF 报告",
                                data=pdf_bytes,
                                file_name=f"cervical_report_{date_str}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                            )
                else:
                    st.warning("安装 `reportlab` 以启用 PDF 导出")
                    st.code("pip install reportlab")


if __name__ == "__main__":
    main()
