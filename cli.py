import argparse
import json
from agent import build_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    result = build_agent().invoke({"question": args.question, "history": []},
                                  config={"recursion_limit": 50})
    print(result["answer"])
    if args.trace:
        print(json.dumps(result.get("trace"), indent=2))
