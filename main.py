import streamlit as st
from pdfsearchengine import PDFSearchEngine
def main():
    # Initialize the PDFSearchApp
    pdfsearchengine = PDFSearchEngine()

    # Set up the Streamlit app layout
    st.title("Multimodal RAG Demo")
    # st.markdown(
    #     "This demo showcases how to use [Colpali](https://github.com/illuin-tech/colpali) embeddings with [Milvus](https://milvus.io/) "
    #     "and utilizing Gemini/OpenAI multimodal RAG for PDF search and Q&A."
    # )

    # Create tabs for Upload PDF and Query
    tab1, tab2 = st.tabs(["Upload PDF", "Query"])

    with tab1:
        # Upload PDF section
        st.header("Upload PDF")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        max_pages = st.slider(
            "Max pages to extract and index",
            min_value=1,
            max_value=50,
            value=20,
            step=10,
        )

        if uploaded_file is not None:
            # Process the uploaded file
            status = pdfsearchengine.process_pdf_upload(
                # session_state={"user_uuid": None},  # Replace with your state logic
                uploaded_file=uploaded_file,
                max_pages=max_pages,
            )
            st.text_area("Indexing Status", value=status, height=100, disabled=True)

    with tab2:
        # Query section
        st.header("Query")
        query = st.text_input("Enter your query")
        if st.button("Query"):
            if query:
                # Perform the search
                image_paths, rag_response = pdfsearchengine.query_pdf(
                    # session_state=,  # Replace with your state logic
                    search_query=query,
                )

                # Display the RAG response
                st.text_area("RAG Response", value=rag_response, height=100, disabled=True)

                # Display the top matching images
                if image_paths:
                    st.header("Top pages matching query")
                    for image_path in image_paths:
                        st.image(image_path, caption=image_path, use_column_width=True)
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()