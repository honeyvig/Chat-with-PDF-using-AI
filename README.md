# Chat-with-PDF-using-AI
Build tool that can analyse PDF / ebooks and answer questions about them. Must be able to retrieve direct quotations.
================
Building a tool that analyzes PDFs or eBooks and answers questions while retrieving direct quotations requires a combination of Natural Language Processing (NLP) and Optical Character Recognition (OCR) if the text is not readily extractable. Here's an outline and Python implementation for this tool:
Step-by-Step Outline

    PDF/Ebook Text Extraction:
        Use libraries like PyPDF2 or pdfplumber for extracting text from PDFs.
        Use OCR tools like Tesseract (via pytesseract) for image-based PDFs.

    Preprocessing:
        Clean and tokenize the text for efficient analysis.

    Question-Answering (QA):
        Use a pre-trained NLP model, like those available in Hugging Face's Transformers library, to answer questions based on the extracted text.

    Direct Quotation Retrieval:
        Implement text search to find and return exact matching excerpts from the text.

    Interface:
        Provide a simple interface (CLI or web-based) to upload PDFs, ask questions, and view answers with quotations.

Python Implementation

Here’s an implementation using pdfplumber for text extraction and Hugging Face’s Transformers library for QA.
Code

import pdfplumber
from transformers import pipeline
from difflib import get_close_matches

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    """Clean and preprocess extracted text."""
    return text.replace('\n', ' ').strip()

def find_closest_quote(context, answer):
    """Find the closest matching text in the context to the answer."""
    lines = context.split('. ')  # Split into sentences.
    matches = get_close_matches(answer, lines, n=1, cutoff=0.5)
    return matches[0] if matches else "No exact match found."

def main():
    # Load pre-trained QA model from Hugging Face
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Upload and extract text
    pdf_path = input("Enter the path to the PDF file: ")
    raw_text = extract_text_from_pdf(pdf_path)
    context = preprocess_text(raw_text)

    print("\nText successfully extracted. You can now ask questions about the document.")

    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting. Thank you!")
            break
        
        # Use QA pipeline to get the answer
        try:
            response = qa_pipeline(question=question, context=context)
            answer = response['answer']
            print(f"\nAnswer: {answer}")

            # Find the closest matching quote
            quote = find_closest_quote(context, answer)
            print(f"Direct Quotation: \"{quote}\"")
        except Exception as e:
            print(f"Error: {e}. Please try again with a different question.")

if __name__ == "__main__":
    main()

How It Works

    Text Extraction:
        The extract_text_from_pdf function reads text from the provided PDF.
        If the PDF contains images, integrate pytesseract for OCR.

    Question Answering:
        The pipeline from Hugging Face runs the QA model on the extracted text.

    Quotation Retrieval:
        The find_closest_quote function uses difflib to find the closest matching sentence in the context to the AI-generated answer.

    Interactive Interface:
        A simple command-line interface allows users to upload a PDF and ask questions.

Requirements

Install the required libraries:

pip install pdfplumber transformers torch pytesseract

For OCR support:

    Install Tesseract OCR: Tesseract Installation Guide.

Further Enhancements

    Web Interface:
        Use a web framework like Flask or FastAPI for a more user-friendly interface.
    Summarization:
        Add text summarization for an overview of the document.
    Handle Large Documents:
        Split large texts into chunks and process them iteratively.
