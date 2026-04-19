import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from generation.generator import Generator
from dotenv import load_dotenv
import os

load_dotenv()

def test_synthesis():
    generator = Generator()
    print(f"Using model: {generator._actual_model} via {generator._actual_provider}")

    # Sample topics and contexts
    samples = [
        {
            "topic": "Abortion Rights",
            "question": "Should a woman have the right to choose whether to have an abortion?",
            "context": [
                {"text": "A woman should have the right to choose whether to have an abortion.", "side": "l", "lean_score": -1.0, "relevance_score": 1.0},
                {"text": "Abortion should be illegal and considered as taking a human life.", "side": "r", "lean_score": 1.0, "relevance_score": 1.0}
            ]
        },
        {
            "topic": "Clean Energy",
            "question": "Are government subsidies for clean energy beneficial?",
            "context": [
                {"text": "Investing in clean energy technologies will create jobs and lead to a more sustainable future.", "side": "l", "lean_score": -1.0, "relevance_score": 1.0},
                {"text": "Government subsidies for clean energy distort the market and waste taxpayer money.", "side": "r", "lean_score": 1.0, "relevance_score": 1.0}
            ]
        },
        {
            "topic": "Universal Basic Income",
            "question": "Is universal basic income necessary to address income inequality?",
            "context": [
                {"text": "Universal basic income is necessary to address income inequality and provide financial security for all citizens.", "side": "l", "lean_score": -1.0, "relevance_score": 1.0},
                {"text": "Universal basic income discourages work and creates dependency on government assistance.", "side": "r", "lean_score": 1.0, "relevance_score": 1.0}
            ]
        }
    ]

    for sample in samples:
        print(f"\n--- Topic: {sample['topic']} ---")
        print(f"Question: {sample['question']}")
        
        result = generator.synthesize(sample['question'], sample['context'])
        
        print(f"Synthesis:\n{result.synthesis}")
        print(f"Word count: {len(result.synthesis.split())}")
        print(f"Sentence count: {result.synthesis.count('.') + result.synthesis.count('!') + result.synthesis.count('?')}")

if __name__ == "__main__":
    test_synthesis()
