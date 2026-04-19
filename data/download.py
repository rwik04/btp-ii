"""
Data loading utilities for twinviews-13k dataset.

Downloads and provides access to the paired liberal/conservative
political text dataset from HuggingFace.

Usage:
    uv run python data/download.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Document:
    """A single document with text, side, and metadata."""

    text: str
    side: str  # 'l' or 'r'
    topic: str
    doc_id: str


@dataclass
class PairedTopic:
    """A topic with paired liberal and conservative documents."""

    topic: str
    left: str
    right: str


def load_twinviews_csv(csv_path: str) -> List[PairedTopic]:
    """
    Load paired documents from a CSV file.

    Expected CSV format:
        l,r,topic
        "liberal text","conservative text","Topic Name"

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of PairedTopic objects
    """
    import csv

    paired: List[PairedTopic] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paired.append(
                PairedTopic(
                    topic=row["topic"],
                    left=row["l"],
                    right=row["r"],
                )
            )
    return paired


def load_twinviews_huggingface(
    dataset_name: str = "wwbrannon/twinviews-13k",
    split: str = "train",
) -> List[PairedTopic]:
    """
    Load paired documents directly from HuggingFace datasets.

    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split (default 'train')

    Returns:
        List of PairedTopic objects
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    paired: List[PairedTopic] = []

    for item in dataset:
        if item.get("l") and item.get("r") and item.get("topic"):
            paired.append(
                PairedTopic(
                    topic=item["topic"],
                    left=item["l"],
                    right=item["r"],
                )
            )

    return paired


def get_unique_topics(paired_docs: List[PairedTopic], sort_by_frequency: bool = False) -> List[str]:
    """
    Extract unique topic strings from paired documents.

    Args:
        paired_docs: List of PairedTopic objects
        sort_by_frequency: If True, sort topics by frequency (descending).
                          Otherwise, sort alphabetically.

    Returns:
        List of unique topic strings
    """
    if sort_by_frequency:
        from collections import Counter
        counts = Counter(d.topic for d in paired_docs)
        # Sort by frequency decending, then alphabetically for ties
        topics = [t for t, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    else:
        topics = sorted({d.topic for d in paired_docs})
    return topics


def split_text(text: str, chunk_size: int = 200, overlap: int = 20) -> List[str]:
    """
    Split a long text into overlapping chunks.

    Uses a simple word-based split to avoid breaking mid-sentence.

    Args:
        text: Input text string
        chunk_size: Target words per chunk
        overlap: Number of overlapping words between chunks

    Returns:
        List of text chunks
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        # Move window forward (step = chunk_size - overlap)
        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size // 2
        start += step

    return chunks


def create_chunked_documents(
    paired_docs: List[PairedTopic],
    chunk_size: int = 200,
    overlap: int = 20,
) -> List[Document]:
    """
    Convert paired topics into chunked documents.

    Each chunk is treated as a separate retrieval unit.

    Args:
        paired_docs: List of PairedTopic objects
        chunk_size: Target words per chunk
        overlap: Overlapping words between chunks

    Returns:
        List of Document objects
    """
    documents: List[Document] = []
    doc_id = 0

    for paired in paired_docs:
        # Chunk left side
        left_chunks = split_text(paired.left, chunk_size, overlap)
        for chunk_text in left_chunks:
            documents.append(
                Document(
                    text=chunk_text,
                    side="l",
                    topic=paired.topic,
                    doc_id=f"{doc_id}",
                )
            )
            doc_id += 1

        # Chunk right side
        right_chunks = split_text(paired.right, chunk_size, overlap)
        for chunk_text in right_chunks:
            documents.append(
                Document(
                    text=chunk_text,
                    side="r",
                    topic=paired.topic,
                    doc_id=f"{doc_id}",
                )
            )
            doc_id += 1

    return documents


def save_paired_topics_csv(paired_docs: List[PairedTopic], output_path: Path) -> None:
    """Save paired topics to CSV with columns: l, r, topic."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["l", "r", "topic"])
        writer.writeheader()
        for item in paired_docs:
            writer.writerow({"l": item.left, "r": item.right, "topic": item.topic})


def main() -> None:
    """Download twinviews from HuggingFace and save it locally as CSV."""
    parser = argparse.ArgumentParser(description="Download twinviews-13k dataset")
    parser.add_argument(
        "--dataset",
        default="wwbrannon/twinviews-13k",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to download",
    )
    parser.add_argument(
        "--out",
        default="data/twinviews-13k.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} (split={args.split})")
    paired_docs = load_twinviews_huggingface(dataset_name=args.dataset, split=args.split)
    topics = get_unique_topics(paired_docs)
    output_path = Path(args.out)
    save_paired_topics_csv(paired_docs, output_path)
    print(f"Loaded {len(paired_docs)} paired rows across {len(topics)} topics.")
    print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    main()
