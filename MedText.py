import os
import pytesseract
from PIL import Image
from transformers import pipeline
import pdfplumber
from llm_text_summarizer import gpt_summarize_medical_report



def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                img = page.to_image(resolution=300)
                ocr_text = pytesseract.image_to_string(img.original)
                text += ocr_text + "\n"
    return text

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def summarize_text(text):
    return gpt_summarize_medical_report(text)

def analyze_report(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".pdf"]:
        text = extract_text_from_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        text = extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type: " + ext)
    text = clean_text(text)
    summary = summarize_text(text)
    return text, summary

if __name__ == "__main__":
    file_path = input("Enter the path to your medical report (PDF/JPG/PNG): ").strip()
    text, summary = analyze_report(file_path)
    print("\n--- Extracted Text ---\n", text)
    print("\n--- Clinical Summary ---\n", summary)


