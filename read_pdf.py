import PyPDF2
import os

pdfs = [
    '545-Article Text-3665-1-10-20241010.pdf',
    'AI2002 Course Outline- Spring 2026.pdf',
    'WhatsApp Image 2026-04-27 at 4.18.58 PM (1).pdf'
]

for pdf in pdfs:
    try:
        reader = PyPDF2.PdfReader(pdf)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        with open(pdf + '.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully read {pdf}")
    except Exception as e:
        print(f"Error reading {pdf}: {e}")
