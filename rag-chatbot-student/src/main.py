"""Command-line chat loop for the local RAG chatbot.

Run:
    python -m src.main
Type 'exit' or press Ctrl+C to quit.

You shouldn't need to change this file - it calls rag_answer() from
src/rag.py, which you'll build in this workshop.
"""
from src.rag import rag_answer


def main():
    print("LumaBox support bot (type 'exit' to quit)")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        if not question:
            continue
        result = rag_answer(question)
        print(f"Bot: {result['answer']}")
        print(f"  (sources: {[h['id'] for h in result['hits']]})")


if __name__ == "__main__":
    main()
