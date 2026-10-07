#frontend/app.py
import os
import requests
import streamlit as st

st.set_page_config(page_title="FinTech Support Assistant", page_icon="💬")
st.title("FinTech Customer Support")
st.caption("Answers are generated from the loaded financial documents. Review the cited sources.")

api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Render historical conversational logs cleanly without duplicate artifacts
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    location = source["file_name"]
                    if source.get("page"):
                        location += f", page {source['page']}"
                    st.markdown(f"**{location}**")
                    st.caption(source["excerpt"])

# 2. Capture and process incoming runtime queries
question = st.chat_input("Ask a question about your financial documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            try:
                response = requests.post(
                    f"{api_url}/chat",
                    json={"question": question},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                answer = result["answer"]
                sources = result["sources"]
            except requests.RequestException as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                answer = f"Unable to reach the assistant: {detail}"
                sources = []
            except (KeyError, ValueError) as exc:
                answer = f"The assistant returned an invalid response: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    location = source["file_name"]
                    if source.get("page"):
                        location += f", page {source['page']}"
                    st.markdown(f"**{location}**")
                    st.caption(source["excerpt"])

    # 3. Store conversation history safely and trigger a rerun to keep the layout updated
    st.session_state.messages.append(
    {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()
