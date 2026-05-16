import os
import argparse
import torch
import pandas as pd
from pypdf import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset


# BERT model that was fine-tuned on SQuAD — good at pulling spans from context paragraphs
QA_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Chunk size to split PDF text into; 500 chars keeps context short enough for BERT's 512 token limit
CHUNK_SIZE = 500


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            all_text += page_text + "\n"
    print(f"Extracted {len(all_text)} characters from {pdf_path}")
    return all_text


def load_qa_pipeline(model_name=QA_MODEL_NAME):
    print(f"Loading QA model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return tokenizer, model, qa_pipeline


def generate_qa_pairs(text, tokenizer, model, qa_pipeline, chunk_size=CHUNK_SIZE):
    qa_pairs = []
    total_chunks = len(range(0, len(text), chunk_size))
    processed = 0

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]
        processed += 1

        if processed % 20 == 0:
            print(f"Processing chunk {processed}/{total_chunks}...")

        # Feed the chunk through BERT to get start/end logits,
        # then decode the span as a generated "question"
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        start_index = torch.argmax(outputs.start_logits)
        end_index = torch.argmax(outputs.end_logits)

        # Decoded span becomes the "question" — not real NLG, but it gives a topical anchor
        question = tokenizer.decode(inputs["input_ids"][0][start_index : end_index + 1])
        question = question.strip()

        if not question or question in ("[CLS]", "[SEP]", ""):
            continue

        # Use the same pipeline to answer the generated question from the same chunk
        try:
            answer = qa_pipeline(question=question, context=chunk)["answer"]
        except Exception:
            continue

        qa_pairs.append({
            "question": question,
            "answer": answer,
            "context": chunk,
        })

    print(f"Generated {len(qa_pairs)} Q&A pairs from {processed} chunks.")
    return qa_pairs


def save_as_csv(qa_pairs, output_csv_path):
    dataset = Dataset.from_list(qa_pairs)
    dataset.to_csv(output_csv_path, index=False)
    print(f"Saved CSV: {output_csv_path}")
    return dataset


def csv_to_jsonl(csv_path, jsonl_path):
    df = pd.read_csv(csv_path)
    df.to_json(jsonl_path, orient="records", lines=True)
    print(f"Saved JSONL: {jsonl_path}")


def show_sample(csv_path, n=5):
    df = pd.read_csv(csv_path)
    print(f"\nSample rows from {csv_path}:")
    print(df.head(n).to_string(index=False))


def pdf_to_qa_dataset(pdf_path, output_csv_path, output_jsonl_path=None, chunk_size=CHUNK_SIZE):
    text = extract_text_from_pdf(pdf_path)
    tokenizer, model, qa_pipeline = load_qa_pipeline()
    qa_pairs = generate_qa_pairs(text, tokenizer, model, qa_pipeline, chunk_size)

    if not qa_pairs:
        print("No Q&A pairs generated. Check if the PDF has readable text.")
        return None

    dataset = save_as_csv(qa_pairs, output_csv_path)

    if output_jsonl_path:
        csv_to_jsonl(output_csv_path, output_jsonl_path)

    return dataset


def convert_existing_csv(csv_path, jsonl_path):
    # If you already have a structured CSV (e.g., manually curated), convert it straight to JSONL
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    csv_to_jsonl(csv_path, jsonl_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a PDF into a structured Q&A dataset (CSV + JSONL) using BERT."
    )
    parser.add_argument("--pdf", default=None, help="Path to the input PDF file.")
    parser.add_argument("--csv_out", default="qa_dataset.csv", help="Output CSV path.")
    parser.add_argument("--jsonl_out", default=None, help="Output JSONL path (optional).")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="Characters per text chunk.")
    parser.add_argument("--from_csv", default=None, help="Skip PDF extraction; convert this existing CSV to JSONL.")
    parser.add_argument("--show_sample", action="store_true", help="Print a few rows after processing.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.from_csv:
        if not args.jsonl_out:
            args.jsonl_out = args.from_csv.replace(".csv", ".jsonl")
        convert_existing_csv(args.from_csv, args.jsonl_out)
        if args.show_sample:
            show_sample(args.from_csv)
        return

    if not args.pdf:
        raise ValueError("Provide --pdf to specify the input PDF, or --from_csv to convert an existing CSV.")

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    dataset = pdf_to_qa_dataset(
        pdf_path=args.pdf,
        output_csv_path=args.csv_out,
        output_jsonl_path=args.jsonl_out,
        chunk_size=args.chunk_size,
    )

    if dataset and args.show_sample:
        show_sample(args.csv_out)


if __name__ == "__main__":
    main()
