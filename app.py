# app.py
import os
import tempfile
from pathlib import Path

import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image

# ── Hugging Face Space details ────────────────────────────────────────────────
TEXT_ONLY_SPACE   = "Samuelbegin/room-image-generation"        # prompt → image
IMAGE_PROMPT_SPACE = "https://samuelbegin-room-decor-ai.hf.space/"  # (img, prompt) → image
API_NAME = "/predict"   # both Spaces expose the same endpoint name
HF_TOKEN = os.getenv("HF_TOKEN")  # leave unset if Spaces are public
# ──────────────────────────────────────────────────────────────────────────────


# Cache the two gradio‑client objects so Streamlit doesn’t reconnect every rerun
@st.cache_resource
def get_clients():
    kwargs = {"hf_token": HF_TOKEN} if HF_TOKEN else {}
    return (
        Client(TEXT_ONLY_SPACE, **kwargs),
        Client(IMAGE_PROMPT_SPACE, **kwargs),
    )

text_client, image_client = get_clients()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Room Interior AI — Generate or Decorate")

uploaded_image = st.file_uploader(
    "Upload a room photo to **decorate** it (PNG/JPG, optional)",
    type=["png", "jpg", "jpeg"],
)
prompt = st.text_input(
    "Prompt",
    value="Dragon blowing fire",
    help="Describe what you want. If you uploaded an image, "
         "this prompt will be used as the decoration instruction.",
)

go = st.button("Generate")

# Optional: show the API schema of the image‑prompt Space for debugging
# with st.expander("Debug: view_api() output"):
#     st.code(image_client.view_api())

# ── Generation logic ─────────────────────────────────────────────────────────
if go:
    with st.spinner("Generating image..."):
        try:
            if uploaded_image is None:
                # 1️⃣  Text‑only generation
                result = text_client.predict(prompt, api_name=API_NAME)
            else:
                # 2️⃣  Image + prompt generation
                # Save the uploaded file to a temp path so handle_file() can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_image.getbuffer())
                    tmp_path = Path(tmp.name)

                result = image_client.predict(
                    handle_file(str(tmp_path)),
                    prompt,
                    api_name=API_NAME,
                )

        except Exception as e:
            st.error(f"❌  Request failed: {e}")
            st.stop()

    # ── Display the returned image (URL or local path) ───────────────────────
    def first_url(r):
        if isinstance(r, str):
            return r
        if isinstance(r, list) and r and isinstance(r[0], str):
            return r[0]
        return None

    image_path = first_url(result)

    if image_path is None:
        st.warning("The Space didn’t return a recognisable image path/URL:")
        st.write(result)
    else:
        # For local paths, open with PIL; for URLs, let Streamlit fetch it
        if image_path.startswith(("http://", "https://")):
            st.image(image_path, caption=prompt)
        elif Path(image_path).exists():
            st.image(Image.open(image_path), caption=prompt)
        else:
            st.warning("Returned path doesn’t exist locally:")
            st.write(image_path)

st.caption("Powered by Samuel Begin ")
