# classifier.py
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Optional, Tuple, Dict

# If Tesseract is not in PATH on Windows, set it explicitly:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

##############################
# Verhoeff checksum (Aadhaar)
##############################
# Tables from standard Verhoeff implementation
_mul = [
 [0,1,2,3,4,5,6,7,8,9],
 [1,2,3,4,0,6,7,8,9,5],
 [2,3,4,0,1,7,8,9,5,6],
 [3,4,0,1,2,8,9,5,6,7],
 [4,0,1,2,3,9,5,6,7,8],
 [5,9,8,7,6,0,4,3,2,1],
 [6,5,9,8,7,1,0,4,3,2],
 [7,6,5,9,8,2,1,0,4,3],
 [8,7,6,5,9,3,2,1,0,4],
 [9,8,7,6,5,4,3,2,1,0],
]

_perm = [
 [0,1,2,3,4,5,6,7,8,9],
 [1,5,7,6,2,8,3,0,9,4],
 [5,8,0,3,7,9,6,1,4,2],
 [8,9,1,6,0,4,3,5,2,7],
 [9,4,5,3,1,2,6,8,7,0],
 [4,2,8,6,5,7,3,9,0,1],
 [2,7,9,3,8,0,6,4,1,5],
 [7,0,4,6,9,1,3,2,5,8],
]

_inv = [0,4,3,2,1,5,6,7,8,9]

def verhoeff_validate(number: str) -> bool:
    """Return True if number (digits-only) passes Verhoeff (checksum==0)."""
    number = re.sub(r'\D', '', number)
    if not number:
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        d = int(ch)
        c = _mul[c][_perm[i % 8][d]]
    return c == 0

##############################
# Image helpers
##############################
def resize_max(image: np.ndarray, max_dim=1000) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / float(max(h, w))
    return cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def find_largest_quad(image: np.ndarray) -> Optional[np.ndarray]:
    """Return 4-point contour approximating largest quadrilateral or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def order_points(pts: np.ndarray) -> np.ndarray:
    # Return points in order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = int(max(widthA, widthB))
    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # optionally increase resolution
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # adaptive threshold to handle lighting
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 15)
    return th

##############################
# OCR + detection
##############################
AADHAAR_REGEXS = [
    re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b'),  # 4-4-4 grouping or contiguous 12
    re.compile(r'\b\d{12}\b')
]
AADHAAR_KEYWORDS = [
    'unique identification authority of india', 'uidai', 'government of india',
    'aadhaar', 'aadhar', 'आधार', 'आधार कार्ड'
]

def extract_aadhaar_like_numbers(text: str) -> list:
    nums = []
    for rx in AADHAAR_REGEXS:
        for m in rx.findall(text):
            digits = re.sub(r'\D', '', m)
            if len(digits) == 12:
                nums.append(digits)
    return list(dict.fromkeys(nums))  # unique

def contains_aadhaar_keywords(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in AADHAAR_KEYWORDS)

def ocr_image_pytesseract(image: np.ndarray, lang: str = 'eng') -> str:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # psm 6 - assume a block of text; adjust if needed
    config = '--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil, lang=lang, config=config)
    return text

##############################
# Main classifier
##############################
class AadhaarClassifier:
    def __init__(self, tesseract_cmd: Optional[str] = None, ocr_lang: str = 'eng'):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.ocr_lang = ocr_lang

    def classify_image(self, image_path: str) -> Dict:
        """Return dict with keys:
           - found_document (bool)
           - ocr_text (str)
           - aadhaar_candidates (list of digits)
           - verified (dict: {number: bool})
           - is_aadhaar_like (bool)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        img = resize_max(img, max_dim=1200)
        quad = find_largest_quad(img)
        if quad is not None:
            warped = four_point_transform(img, quad)
            processed = enhance_for_ocr(warped)
            # convert back to BGR for pytesseract path
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            text = ocr_image_pytesseract(processed_bgr, lang=self.ocr_lang)
            doc_found = True
        else:
            # fallback: OCR on whole image (maybe top of photo)
            processed = enhance_for_ocr(img)
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            text = ocr_image_pytesseract(processed_bgr, lang=self.ocr_lang)
            doc_found = False

        candidates = extract_aadhaar_like_numbers(text)
        verified = {num: verhoeff_validate(num) for num in candidates}
        aadhaar_like = contains_aadhaar_keywords(text) or len(candidates) > 0

        return {
            "found_document": doc_found,
            "ocr_text": text,
            "aadhaar_candidates": candidates,
            "verified": verified,
            "is_aadhaar_like": aadhaar_like
        }

if __name__ == "__main__":
    import argparse, json, sys
    parser = argparse.ArgumentParser(description="Aadhaar image classifier (Tesseract)")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--tesseract", help="Path to tesseract cmd", default=None)
    parser.add_argument("--lang", help="OCR language (default eng)", default="eng")
    args = parser.parse_args()
    ac = AadhaarClassifier(tesseract_cmd=args.tesseract, ocr_lang=args.lang)
    try:
        result = ac.classify_image(args.image)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(2)
