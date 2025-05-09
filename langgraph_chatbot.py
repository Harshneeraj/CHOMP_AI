from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,  HumanMessage

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key="")


class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chat_bot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chat_bot", chat_bot)
graph_builder.add_edge(START, "chat_bot")
graph = graph_builder.compile()

# Interactive loop
if __name__ == "__main__":
    messages = [SystemMessage(content="You are a helpful assistant.")]
    
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "end":
            print("Assistant: Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))
        final_state = graph.invoke({"messages": messages})
        reply = final_state["messages"][-1]
        messages.append(reply)
        print(f"Assistant: {reply.content}")