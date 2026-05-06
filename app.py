"""
Handwritten Telugu Vowel Recognition — Streamlit App
SMAI Assignment 3 | Tier 1 | Variant 3.4

Draw a Telugu vowel on the canvas and the model will predict which vowel it is.
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import random

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Telugu Vowel Recognizer",
    page_icon="✍️",
    layout="wide",
)

# ─── Model Definition (must match training notebook exactly) ─
IMG_SIZE = 64


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> MaxPool"""
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TeluguCNN(nn.Module):
    """3-layer CNN for Telugu vowel classification (~300k params)."""
    def __init__(self, num_blocks=3, base_filters=16, dropout_rate=0.25, num_classes=6):
        super().__init__()
        self.num_blocks = num_blocks

        blocks = []
        in_ch = 1
        out_ch = base_filters
        for i in range(num_blocks):
            blocks.append(ConvBlock(in_ch, out_ch, pool=True))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 128)

        self.features = nn.Sequential(*blocks)

        spatial = IMG_SIZE // (2 ** num_blocks)
        flat_size = in_ch * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(flat_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    ckpt = torch.load("telugu_vowel_model.pth", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = TeluguCNN(
        num_blocks=cfg["num_blocks"],
        base_filters=cfg["base_filters"],
        dropout_rate=cfg["dropout_rate"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


try:
    model, ckpt = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─── Telugu Vowel Info ───────────────────────────────────────
TELUGU_MAP = {
    "A": "\u0C05",   # అ
    "Aa": "\u0C06",  # ఆ
    "Ai": "\u0C0E",  # ఎ
    "E": "\u0C07",   # ఇ
    "Ee": "\u0C08",  # ఈ
    "U": "\u0C09",   # ఉ
}

CLASS_NAMES = sorted(TELUGU_MAP.keys())

# ─── Preprocessing ───────────────────────────────────────────
def preprocess_canvas(canvas_result):
    """Convert canvas drawing to model input tensor."""
    img_data = canvas_result.image_data
    img = Image.fromarray(img_data.astype("uint8"), "RGBA")
    img_gray = img.convert("L")
    img_gray = ImageOps.invert(img_gray)

    img_array = np.array(img_gray)
    coords = np.argwhere(img_array > 20)
    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    pad = 20
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(img_array.shape[0], y_max + pad)
    x_max = min(img_array.shape[1], x_max + pad)

    cropped = img_gray.crop((x_min, y_min, x_max, y_max))

    w, h = cropped.size
    max_dim = max(w, h)
    square = Image.new("L", (max_dim, max_dim), 0)
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    square.paste(cropped, (paste_x, paste_y))

    resized = square.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    tensor = torch.FloatTensor(np.array(resized, dtype=np.float32) / 255.0)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def run_inference(canvas_result):
    """Run model inference on canvas result. Returns (predicted_name, confidence, probabilities) or None."""
    if canvas_result.image_data is None:
        return None
    input_tensor = preprocess_canvas(canvas_result)
    if input_tensor is None:
        return None
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1).squeeze().numpy()
        predicted_idx = probabilities.argmax()
        predicted_name = CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx] * 100
    return predicted_name, confidence, probabilities, input_tensor


# ─── Session State Init ──────────────────────────────────────
if "practice_target" not in st.session_state:
    st.session_state.practice_target = random.choice(CLASS_NAMES)
if "practice_score" not in st.session_state:
    st.session_state.practice_score = 0
if "practice_attempts" not in st.session_state:
    st.session_state.practice_attempts = 0
if "practice_feedback" not in st.session_state:
    st.session_state.practice_feedback = None  # None | "correct" | "wrong" | "unclear"
if "practice_canvas_key" not in st.session_state:
    st.session_state.practice_canvas_key = 0  # increment to reset canvas


# ─── UI ──────────────────────────────────────────────────────
st.title("✍️ Handwritten Telugu Vowel Recognition")
st.markdown(
    "Draw a Telugu vowel character on the canvas below. "
    "The CNN model will predict which of the *6 vowels* it is."
)

if not model_loaded:
    st.error(
        "*Model file not found!* Please place `telugu_vowel_model.pth` "
        "in the same directory as this app."
    )
    st.stop()

# ─── Mode Selector ───────────────────────────────────────────
mode = st.radio(
    "Mode",
    ["🔍 Free Draw", "🎯 Practice Mode"],
    horizontal=True,
)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# FREE DRAW MODE
# ════════════════════════════════════════════════════════════
if mode == "🔍 Free Draw":

    # Display model info
    with st.expander("Model Info", expanded=False):
        cfg = ckpt["config"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Architecture", f"{cfg['num_blocks']}-layer CNN")
        col2.metric("Parameters", f"~{sum(p.numel() for p in model.parameters()):,}")
        col3.metric(
            "Test Accuracy",
            f"{ckpt.get('test_accuracy', 'N/A'):.1f}%"
            if isinstance(ckpt.get("test_accuracy"), (int, float))
            else "N/A",
        )
        st.markdown("*Ablation Config:*")
        st.json(cfg)

    col_canvas, col_result = st.columns([1, 1], gap="large")

    with col_canvas:
        st.subheader("Draw Here")
        st.caption("Use your mouse or touch to draw a Telugu vowel character.")

        stroke_width = st.slider("Stroke Width", 8, 20, 10, key="stroke_free")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="canvas_free",
        )

        st.markdown("---")
        st.markdown("*Reference Telugu Vowels:*")
        vowel_cols = st.columns(6)
        for i, name in enumerate(CLASS_NAMES):
            with vowel_cols[i]:
                st.markdown(
                    f"<div style='text-align:center; font-size:2rem;'>{TELUGU_MAP[name]}</div>"
                    f"<div style='text-align:center; font-size:0.8rem; color:gray;'>{name}</div>",
                    unsafe_allow_html=True,
                )

    with col_result:
        st.subheader("Prediction")
        result = run_inference(canvas_result) if canvas_result.image_data is not None else None

        if result is not None:
            predicted_name, confidence, probabilities, input_tensor = result

            st.markdown("*Preprocessed Input (64×64):*")
            preprocessed_img = input_tensor.squeeze().numpy() * 0.5 + 0.5
            st.image(preprocessed_img, width=128, clamp=True)

            st.markdown("---")
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span style='font-size:5rem;'>{TELUGU_MAP[predicted_name]}</span><br>"
                f"<span style='font-size:1.5rem; font-weight:bold;'>{predicted_name}</span><br>"
                f"<span style='font-size:1.2rem; color:{'green' if confidence > 80 else 'orange' if confidence > 50 else 'red'};'>"
                f"{confidence:.1f}% confidence</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("*Class Probabilities:*")
            import pandas as pd

            prob_df = pd.DataFrame({
                "Vowel": [f"{name} ({TELUGU_MAP[name]})" for name in CLASS_NAMES],
                "Probability (%)": probabilities * 100,
            })
            prob_df = prob_df.sort_values("Probability (%)", ascending=True)
            st.bar_chart(prob_df.set_index("Vowel"), horizontal=True)
        else:
            st.info("👆 Draw a character on the canvas to see predictions.")


# ════════════════════════════════════════════════════════════
# PRACTICE MODE
# ════════════════════════════════════════════════════════════
else:
    target = st.session_state.practice_target
    target_char = TELUGU_MAP[target]
    score = st.session_state.practice_score
    attempts = st.session_state.practice_attempts

    # ── Score Header ────────────────────────────────────────
    score_col, _, reset_col = st.columns([2, 3, 1])
    with score_col:
        accuracy = (score / attempts * 100) if attempts > 0 else 0
        st.markdown(
            f"<div style='font-size:1.1rem;'>✅ <b>{score}</b> correct &nbsp;|&nbsp; "
            f"🎯 <b>{attempts}</b> attempts &nbsp;|&nbsp; "
            f"📊 <b>{accuracy:.0f}%</b> accuracy</div>",
            unsafe_allow_html=True,
        )
    with reset_col:
        if st.button("🔄 Reset Score"):
            st.session_state.practice_score = 0
            st.session_state.practice_attempts = 0
            st.session_state.practice_feedback = None
            st.session_state.practice_target = random.choice(CLASS_NAMES)
            st.session_state.practice_canvas_key += 1
            st.rerun()

    st.markdown("---")

    col_target, col_canvas, col_feedback = st.columns([1, 1.2, 1], gap="large")

    # ── Target Character Panel ───────────────────────────────
    with col_target:
        st.subheader("✏️ Your Target")
        st.markdown(
            f"<div style='"
            f"text-align:center; "
            f"background: linear-gradient(135deg, #1a1a2e, #16213e); "
            f"border-radius: 16px; "
            f"padding: 32px 16px; "
            f"border: 2px solid #4a4a8a;'>"
            f"<div style='font-size:6rem; line-height:1.1;'>{target_char}</div>"
            f"<div style='font-size:1.4rem; color:#a0a0d0; margin-top:8px; font-weight:600;'>{target}</div>"
            f"<div style='font-size:0.85rem; color:#606090; margin-top:4px;'>Draw this character →</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Quick reference for all vowels
        st.markdown("---")
        st.markdown("**Reference:**")
        for name in CLASS_NAMES:
            highlight = "**" if name == target else ""
            marker = " ← target" if name == target else ""
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:10px; "
                f"padding: 4px 8px; border-radius:8px; "
                f"background: {'rgba(100,100,200,0.15)' if name == target else 'transparent'};'>"
                f"<span style='font-size:1.6rem;'>{TELUGU_MAP[name]}</span>"
                f"<span style='font-size:0.9rem; color:{'#a0a0ff' if name == target else '#888'};'>"
                f"{'<b>' if name == target else ''}{name}{'</b>' if name == target else ''}"
                f"{'<em>  ← target</em>' if name == target else ''}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Canvas Panel ─────────────────────────────────────────
    with col_canvas:
        st.subheader("🎨 Draw Here")
        stroke_width = st.slider("Stroke Width", 8, 20, 12, key="stroke_practice")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key=f"canvas_practice_{st.session_state.practice_canvas_key}",
        )

        # Action buttons
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            check_pressed = st.button("✅ Check", use_container_width=True, type="primary")
        with b_col2:
            skip_pressed = st.button("⏭️ Skip", use_container_width=True)

    # ── Feedback Panel ───────────────────────────────────────
    with col_feedback:
        st.subheader("📊 Feedback")

        # Handle Check button
        if check_pressed:
            result = run_inference(canvas_result) if canvas_result.image_data is not None else None
            if result is None:
                st.session_state.practice_feedback = {
                    "kind": "empty",
                    "predicted": None,
                    "confidence": None,
                    "probabilities": None,
                }
            else:
                predicted_name, confidence, probabilities, _ = result
                st.session_state.practice_attempts += 1

                if predicted_name == target and confidence >= 50:
                    kind = "correct"
                    st.session_state.practice_score += 1
                elif predicted_name == target and confidence < 50:
                    kind = "unclear"
                else:
                    kind = "wrong"

                st.session_state.practice_feedback = {
                    "kind": kind,
                    "predicted": predicted_name,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }

        # Handle Skip button
        if skip_pressed:
            st.session_state.practice_target = random.choice(CLASS_NAMES)
            st.session_state.practice_feedback = None
            st.session_state.practice_canvas_key += 1
            st.rerun()

        # Render feedback
        fb = st.session_state.practice_feedback

        if fb is None:
            st.markdown(
                "<div style='text-align:center; color:#666; padding:40px 0;'>"
                "<div style='font-size:3rem;'>🖊️</div>"
                "<div>Draw the target character<br>then press <b>Check</b>.</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        elif fb["kind"] == "empty":
            st.warning("Canvas is empty — draw something first!")

        elif fb["kind"] == "correct":
            st.markdown(
                "<div style='text-align:center; padding:20px; "
                "background: linear-gradient(135deg, #0d2b1a, #0f3d24); "
                "border-radius:16px; border: 2px solid #2ecc71;'>"
                "<div style='font-size:3rem;'>🎉</div>"
                "<div style='font-size:1.5rem; color:#2ecc71; font-weight:bold;'>Correct!</div>"
                f"<div style='color:#aaa; margin-top:4px;'>Confidence: {fb['confidence']:.1f}%</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            # Next character button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▶️ Next Character", use_container_width=True, type="primary"):
                st.session_state.practice_target = random.choice(CLASS_NAMES)
                st.session_state.practice_feedback = None
                st.session_state.practice_canvas_key += 1
                st.rerun()

        elif fb["kind"] == "unclear":
            st.markdown(
                "<div style='text-align:center; padding:20px; "
                "background: linear-gradient(135deg, #1a1500, #2b2200); "
                "border-radius:16px; border: 2px solid #f39c12;'>"
                "<div style='font-size:3rem;'>🤔</div>"
                "<div style='font-size:1.4rem; color:#f39c12; font-weight:bold;'>Almost!</div>"
                f"<div style='color:#aaa; margin-top:4px;'>Model recognised <b>{TELUGU_MAP[fb['predicted']]} ({fb['predicted']})</b> "
                f"but with low confidence ({fb['confidence']:.1f}%).<br>Try drawing more clearly!</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔁 Try Again", use_container_width=True):
                st.session_state.practice_feedback = None
                st.session_state.practice_canvas_key += 1
                st.rerun()

        elif fb["kind"] == "wrong":
            st.markdown(
                "<div style='text-align:center; padding:20px; "
                "background: linear-gradient(135deg, #2b0d0d, #3d1010); "
                "border-radius:16px; border: 2px solid #e74c3c;'>"
                "<div style='font-size:3rem;'>❌</div>"
                "<div style='font-size:1.4rem; color:#e74c3c; font-weight:bold;'>Not quite!</div>"
                f"<div style='color:#aaa; margin-top:6px;'>"
                f"You drew: <span style='font-size:1.4rem;'>{TELUGU_MAP[fb['predicted']]}</span> "
                f"<b>{fb['predicted']}</b> ({fb['confidence']:.1f}%)<br>"
                f"Target was: <span style='font-size:1.4rem;'>{target_char}</span> <b>{target}</b>"
                f"</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            retry_col, next_col = st.columns(2)
            with retry_col:
                if st.button("🔁 Try Again", use_container_width=True):
                    st.session_state.practice_feedback = None
                    st.session_state.practice_canvas_key += 1
                    st.rerun()
            with next_col:
                if st.button("▶️ Next", use_container_width=True):
                    st.session_state.practice_target = random.choice(CLASS_NAMES)
                    st.session_state.practice_feedback = None
                    st.session_state.practice_canvas_key += 1
                    st.rerun()

        # Show probability bars after any check
        if fb is not None and fb["kind"] != "empty" and fb.get("probabilities") is not None:
            st.markdown("---")
            st.markdown("**Class Probabilities:**")
            import pandas as pd
            prob_df = pd.DataFrame({
                "Vowel": [f"{name} ({TELUGU_MAP[name]})" for name in CLASS_NAMES],
                "Probability (%)": fb["probabilities"] * 100,
            })
            prob_df = prob_df.sort_values("Probability (%)", ascending=True)
            st.bar_chart(prob_df.set_index("Vowel"), horizontal=True)


# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "SMAI Assignment 3 | Tier 1 | Variant 3.4 — "
    "Handwritten Indic Script Recognition (Telugu Vowels)"
)