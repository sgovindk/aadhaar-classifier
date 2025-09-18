"""
Streamlit frontend for Aadhaar classifier

Usage:
  1. Make sure your virtualenv is active and you have installed dependencies:
     pip install streamlit opencv-python pillow pytesseract numpy

  2. Ensure Tesseract is installed and in PATH (or edit tesseract_cmd below).

  3. Put this file in the same folder as your `classifier.py` (the module
     that defines AadhaarClassifier). Run:

     streamlit run streamlit_app.py

This app provides:
 - file upload (image)
 - language selection (eng, hin, eng+hin)
 - run classifier and show JSON results
 - quick Tesseract debug (try different PSMs and show text)
 - display original image
 - option to download results as JSON
"""

import streamlit as st
from PIL import Image
import pytesseract
import json
import os
import tempfile
from classifier import AadhaarClassifier

# Optional: if Tesseract not in PATH, uncomment and set path
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

st.set_page_config(page_title="Aadhaar Classifier", layout="centered")
st.title("Aadhaar Card Classifier — Streamlit UI")
st.write("Upload an image of an Aadhaar card and the app will run OCR + Verhoeff check.")

# Sidebar options
st.sidebar.header("Options")
lang = st.sidebar.selectbox("OCR language", options=["eng", "hin", "eng+hin"], index=0)
run_debug = st.sidebar.checkbox("Show Tesseract debug (psm samples)", value=True)

uploaded = st.file_uploader("Upload Aadhaar image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to get started. You can also drag & drop files.")
else:
    # Display uploaded image
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not open the uploaded file: {e}")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    # Save temp file for classifier which expects a path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        image.save(temp_path, format="JPEG")

    st.write("---")
    st.subheader("Classifier run")
    ac = AadhaarClassifier(ocr_lang=lang)

    try:
        with st.spinner("Running classifier..."):
            result = ac.classify_image(temp_path)
    except Exception as e:
        st.error(f"Classifier error: {e}")
        try:
            os.unlink(temp_path)
        except Exception:
            pass
        st.stop()

    st.success("Classifier finished")

    # Show structured result
    st.markdown("**Result (parsed)**")
    st.json(result)

    # OCR raw text
    st.markdown("**OCR raw text**")
    st.text_area("OCR Text", value=result.get("ocr_text", ""), height=180)

    # Aadhaar candidates
    candidates = result.get("aadhaar_candidates", [])
    verified = result.get("verified", {})
    if candidates:
        st.markdown("**Detected Aadhaar-like numbers**")
        for c in candidates:
            ok = verified.get(c, False)
            status = "✅ valid (Verhoeff)" if ok else "❌ invalid checksum"
            st.write(f"{c} — {status}")
    else:
        st.info("No 12-digit Aadhaar-like numbers were detected.")

    # Option to download JSON
    st.download_button("Download result JSON",
                       json.dumps(result, indent=2, ensure_ascii=False),
                       file_name="aadhaar_result.json",
                       mime="application/json")

    st.write("---")
    if run_debug:
        st.subheader("Tesseract debug (different PSMs)")
        pil = image
        psm_values = [3, 6, 7, 11]
        for psm in psm_values:
            cfg = f"--oem 3 --psm {psm}"
            try:
                text = pytesseract.image_to_string(pil, lang=lang, config=cfg)
            except Exception as e:
                text = f"Tesseract error: {e}"
            st.markdown(f"**PSM {psm}**")
            st.text_area(f"PSM {psm} output", value=text, height=140)

    # cleanup temp file
    try:
        os.unlink(temp_path)
    except Exception:
        pass

    st.write("---")
    st.caption("Tip: If OCR results are poor, try taking a higher-resolution photo, "
               "improving lighting, or cropping so the card fills the frame.")
