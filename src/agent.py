from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.prompt import system_prompt


FALLBACK = (
    "I could not find this information in the clinic documents. "
    "Please contact the clinic staff for confirmation."
)

EMERGENCY_MSG = (
    "⚠️ This sounds like a medical emergency. Please call your local emergency "
    "number right now (in India: 112 or 108) or go to the nearest emergency "
    "room immediately. I can't handle emergencies here."
)

UNSAFE_MSG = (
    "I'm not able to give diagnoses, prescriptions, or medication/dosage advice. "
    "Please consult a qualified doctor for that. I can help with clinic info like "
    "appointments, timings, fees, doctors, lab tests, and policies."
)


# ----------------------------------------------------------------------
# Graph state
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation (memory)
    route: str                                 # guardrail decision
    context: str                               # retrieved docs


# ----------------------------------------------------------------------
# Build the compiled agent
# ----------------------------------------------------------------------
def build_medical_agent(retriever, llm):

    # --- Retriever exposed as a TOOL ---
    @tool
    def clinic_document_search(query: str) -> str:
        """Search the clinic's medical documents for relevant information.
        Use for questions about appointments, doctor availability, consultation
        fees, clinic timings, lab test instructions, medicine refill policy,
        report collection, and general clinic guidance."""
        docs = retriever.invoke(query)
        if not docs:
            return "NO_CONTEXT"
        return "\n\n".join(d.page_content for d in docs)

    # --- Node 1: guardrail (safety classifier) ---
    def guardrail(state: AgentState):
        user_msg = state["messages"][-1].content
        prompt = [
            SystemMessage(content=(
                "You are a safety classifier for a medical clinic assistant. "
                "Classify the user's message into EXACTLY one label:\n"
                "EMERGENCY = a medical emergency (chest pain, severe bleeding, "
                "trouble breathing, unconscious, stroke signs, or self-harm/suicide intent).\n"
                "UNSAFE = asks for a diagnosis, a prescription, a drug dosage to take, "
                "or to start/stop/change medication.\n"
                "SAFE = a normal informational question about the clinic (appointments, "
                "timings, fees, doctors, lab tests, policies, reports).\n"
                "Reply with ONLY one word: EMERGENCY, UNSAFE, or SAFE."
            )),
            HumanMessage(content=user_msg),
        ]
        label = llm.invoke(prompt).content.strip().upper()
        if "EMERGENCY" in label:
            route = "emergency"
        elif "UNSAFE" in label:
            route = "unsafe"
        else:
            route = "safe"
        return {"route": route}

    def route_after_guardrail(state: AgentState) -> Literal["emergency", "unsafe", "retrieve"]:
        return "retrieve" if state["route"] == "safe" else state["route"]

    # --- Node 2a / 2b: canned safe responses ---
    def emergency_response(state: AgentState):
        return {"messages": [AIMessage(content=EMERGENCY_MSG)]}

    def unsafe_response(state: AgentState):
        return {"messages": [AIMessage(content=UNSAFE_MSG)]}

    # --- Node 3: retrieve (with follow-up reformulation) ---
    def retrieve(state: AgentState):
        messages = state["messages"]
        query = messages[-1].content

        # If there is prior history, rewrite follow-ups into a standalone question
        if len(messages) > 1:
            reformulate = [
                SystemMessage(content=(
                    "Given the chat history and the latest user question, rewrite the "
                    "question so it is fully self-contained (resolve pronouns like 'she', "
                    "'it', 'that'). Return ONLY the rewritten question, nothing else."
                )),
            ] + messages
            query = llm.invoke(reformulate).content.strip()

        context = clinic_document_search.invoke({"query": query})
        return {"context": context}

    # --- Node 4: generate ---
    def generate(state: AgentState):
        context = state.get("context", "")
        if not context or context == "NO_CONTEXT":
            return {"messages": [AIMessage(content=FALLBACK)]}

        # Inject context into the system prompt (handles {context} placeholder if present)
        if "{context}" in system_prompt:
            sys_text = system_prompt.replace("{context}", context)
        else:
            sys_text = f"{system_prompt}\n\nContext from clinic documents:\n{context}"

        msgs = [SystemMessage(content=sys_text)] + state["messages"]
        answer = llm.invoke(msgs)
        return {"messages": [answer]}

    # --- Wire the graph ---
    builder = StateGraph(AgentState)
    builder.add_node("guardrail", guardrail)
    builder.add_node("emergency", emergency_response)
    builder.add_node("unsafe", unsafe_response)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"emergency": "emergency", "unsafe": "unsafe", "retrieve": "retrieve"},
    )
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    builder.add_edge("emergency", END)
    builder.add_edge("unsafe", END)

    # Multi-turn memory (in-process). For persistence across restarts,
    # swap MemorySaver for SqliteSaver / PostgresSaver later.
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ----------------------------------------------------------------------
# Helper: one call = one user turn
# ----------------------------------------------------------------------
def ask_agent(graph, question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return result["messages"][-1].content