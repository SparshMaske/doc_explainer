# backend/ocr_utils.py
import io, os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from pdfminer.high_level import extract_text


# Try to extract embedded text first, then fallback to OCR


def extract_text_from_pdf(path: str) -> str:
try:
text = extract_text(path)
if text and text.strip():
return text
except Exception:
pass
# fallback to OCR pages
return ocr_pdf_images(path)




def ocr_pdf_images(path: str) -> str:
texts = []
pages = convert_from_path(path, dpi=200)
for page in pages:
text = pytesseract.image_to_string(page)
texts.append(text)
return "\n\n".join(texts)




def ocr_image_bytes(file_bytes: bytes) -> str:
img = Image.open(io.BytesIO(file_bytes))
return pytesseract.image_to_string(img)
