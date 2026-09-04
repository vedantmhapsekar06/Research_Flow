from services.chat_service import answer_question

result = answer_question("What is BM25 used for in this paper?")
print("ANSWER:\n", result["answer"])
print("\nSOURCES:")
for s in result["sources"]:
    print(f"- {s['paper_name']} (page {s['page']}): {s['snippet'][:80]}...")

print("\n--- Follow-up ---")
result2 = answer_question("How much did it improve accuracy by?")
print("ANSWER:\n", result2["answer"])