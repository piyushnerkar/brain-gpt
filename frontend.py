import streamlit as st
import requests

# Set your backend URL base
BACKEND_URL = "https://brain-gpt.onrender.com"

st.title("🧠 Docu-Query AI Assistant")
st.subheader("FastAPI + LangChain + Streamlit RAG Pipeline")

# --- Section 1: Data Ingestion ---
if st.button("Load Knowledge Base"):
    with st.spinner("Processing PDFs and indexing vectors..."):
        try:
            response = requests.post(f"{BACKEND_URL}/load-knowledge-base")
            response.raise_for_status() 
            data = response.json()
            
            # Error checking based on custom dictionary return
            if data.get("status") == "Error":
                st.error(data.get("message"))
            elif "status" in data:
                st.success(f"🎉 {data['status']} ({data.get('pages', 0)} pages processed)")
            else:
                st.info("Knowledge base loaded successfully!")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.write("---")

# --- Section 2: Q&A Interface ---
user_question = st.text_input("Ask a question about your documents:")
if st.button("Ask"):
    if user_question.strip() == "":
        st.warning("Please type a valid question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask", 
                    json={"question": user_question}
                )
                response.raise_for_status()
                data = response.json()
                
                st.markdown("### Answer:")
                st.write(data.get("answer", "No response key found in backend data."))
                
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch answer from backend: {e}")