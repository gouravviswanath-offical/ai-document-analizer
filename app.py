import streamlit as st
import tempfile
import os
import sys

# Add subdirectories to path because of the nested project structure
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(script_dir), "vector_store.py"))
sys.path.append(os.path.join(os.path.dirname(script_dir), "agent.py"))

from vector_store import create_vector_store
from agent import ask_ai

st.set_page_config(page_title="Document Chat AI", layout="centered")

st.title("📄 Engineering Document Chat AI")
st.write("Upload your PDF manual and ask the AI questions about it.")

uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Processing document and generating embeddings..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                create_vector_store(tmp_path)
                st.success("Document processed successfully! You can now ask questions below.")
            except Exception as e:
                st.error(f"Error processing document: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass # Windows file lock issue might prevent deletion immediately

st.divider()

st.subheader("Ask Questions")
question = st.text_input("Enter your question about the uploaded document:")

if st.button("Ask AI"):
    if question:
        if not os.path.exists("engineering_db"):
            st.warning("Please upload and process a document first before asking questions.")
        else:
            with st.spinner("Wait for it... Thinking..."):
                try:
                    answer = ask_ai(question)
                    st.markdown("### Answer")
                    st.info(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
    else:
        st.warning("Please enter a question first.")
