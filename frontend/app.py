"""Streamlit chat client for the FinTech Customer Support Assistant."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
HISTORY_TURNS = 6

st.set_page_config(page_title="FinTech Support Assistant", page_icon="💬")
st.title("FinTech Customer Support")
st.caption("Answers come only from the loaded FinBase documents. Every answer lists the sources it used.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"API: {API_URL}")


def render_sources(sources: list[dict]) -> None:
    cited = [s for s in sources if s["cited"]]
    other = [s for s in sources if not s["cited"]]
    if cited:
        with st.expander(f"Sources used ({len(cited)})", expanded=False):
            for source in cited:
                st.markdown(f"**{source['label']}**")
                st.caption(f"Similarity {source['similarity']:.2f} · {source['file_name']}")
                st.text(source["excerpt"])
    if other:
        with st.expander(f"Other retrieved passages ({len(other)})", expanded=False):
            for source in other:
                st.markdown(f"**{source['label']}**")
                st.caption(f"Similarity {source['similarity']:.2f}")


def render_assistant(message: dict) -> None:
    st.markdown(message["content"])
    meta = message.get("meta")
    if not meta:
        return
    if meta["refused"]:
        st.caption("Not found in the knowledge base.")
    else:
        st.caption(f"Retrieval confidence {meta['retrieval_score']:.2f} · mode: {meta['retrieval_mode']}")
        if not meta["citations_verified"] and meta["route"] == "rag":
            st.warning("The citations in this answer could not be verified. Please check the sources below.")
    if meta["standalone_question"] and meta["standalone_question"] != message.get("question"):
        st.caption(f"Interpreted as: {meta['standalone_question']}")
    if message.get("sources"):
        render_sources(message["sources"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message)
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask a question about FinBase products and policies")
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-HISTORY_TURNS:]]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat", json={"question": question, "history": history}, timeout=120
                )
                response.raise_for_status()
                result = response.json()
                reply = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "question": question,
                    "meta": {
                        key: result[key]
                        for key in ("refused", "route", "retrieval_score", "citations_verified", "standalone_question", "retrieval_mode")
                    },
                }
            except requests.RequestException as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                reply = {"role": "assistant", "content": f"Unable to reach the assistant: {detail}", "sources": []}
            except (KeyError, ValueError) as exc:
                reply = {"role": "assistant", "content": f"The assistant returned an invalid response: {exc}", "sources": []}
        render_assistant(reply)
    st.session_state.messages.append(reply)
