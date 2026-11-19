"""
An end-to-end application testing. Unit test cases not covered.
"""

from graph import graph

TEST_CASES = [
    ("off_topic", "eval_1", "best masala dosa in bluru", ("cannot answer",)),
    ("table_lookup", "eval_2", "What are the exact suggested actions and GAI risks listed under GOVERN 1.1?", ("gv-1.1-001",)),
    ("follow_up", "eval_2", "Which AI actors are responsible for implementing those actions?", ("governance and oversight", "senior leadership", "board of directors")),
    ("out_of_scope_ai", "eval_3", "What exact fines or penalties does the EU AI Act impose for non-compliance?", ("cannot answer",)),
]


def run_eval():
    config = {"configurable": {"thread_id": "eval_session"}}

    for name, thread_id, query, expected_terms in TEST_CASES:
        result = graph.invoke(
            {
                "query": query,
                "retry_count": 0,
                "relevant_threshold": 1,
                "chunks": [],
                "is_relevant": False,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1].content
        passed = any(term in answer.lower() for term in expected_terms)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: '{query}'")
        if not passed:
            print(f"   Expected one of {expected_terms} in response, got:\n   {answer}\n")


if __name__ == "__main__":
    run_eval()