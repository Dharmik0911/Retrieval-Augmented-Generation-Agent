import json
from pathlib import Path
from agent import build_agent

QUESTIONS = [
    "What was the probable cause of the VT-PTE accident?",
    "Compare contributing factors for VT-PTE and VT-GDI.",
    "What findings in VT-ANB are preliminary?",
    "What caused the VT-ZZZ accident?",
    "Hello",
]

if __name__ == "__main__":
    agent = build_agent()
    results = []
    for question in QUESTIONS:
        try:
            result = agent.invoke({"question": question, "history": []},
                                  config={"recursion_limit": 50})
            results.append({"question": question, "answer": result["answer"],
                            "trace": result.get("trace"), "model_self_check": result.get("grounded"),
                            "reviewer_correctness": None, "reviewer_citation_support": None})
        except Exception as exc:
            results.append({"question": question, "error": type(exc).__name__})
    folder = Path(__file__).resolve().parent / "runs"
    folder.mkdir(exist_ok=True)
    (folder / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {len(results)} cases; manually verify against PDF evidence.")
