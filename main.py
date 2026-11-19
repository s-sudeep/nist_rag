from graph import graph


def run_chat():
    config = {"configurable": {"thread_id": "demo_session"}}
    print("NIST AI RMF Assistant ready. Type 'quit' to exit.")

    while True:
        user_input = input("\nAsk NIST AI RMF: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        result = graph.invoke(
            {
                "query": user_input,
                "retry_count": 0,
                "relevant_threshold": 1,
                "chunks": [],
                "is_relevant": False,
            },
            config=config,
        )
        print("\nAssistant:\n", result["messages"][-1].content)


if __name__ == "__main__":
    run_chat()