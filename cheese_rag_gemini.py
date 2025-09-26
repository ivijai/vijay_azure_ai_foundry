import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import gradio as gr
import re
from typing import List
from dotenv import load_dotenv
import os

# Load environment variables (create a .env file with your API key)
load_dotenv()

# Configuration
# Load API key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError(
        "Error: GOOGLE_API_KEY is not set. Please add it to your .env file.")

EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model
# GENERATION_MODEL = "models/gemini-1.5-pro"  # Gemini generation model
GENERATION_MODEL = "models/gemini-2.0-flash-001"
CHROMA_DB_NAME = "cheese_rag_db_gemini"

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Document Processing


def process_book_text(text: str) -> List[str]:
    """Split the book text into meaningful chunks"""
    # Remove page markers
    text = re.sub(r"===== Page \d+ =====\n", "", text)

    # Split by sections (headings in all caps or bold)
    sections = re.split(r"\n\s*#+|\n\s*\*\*", text)

    # Filter and clean sections
    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) > 100:  # Only keep substantial sections
            # Clean up formatting
            section = re.sub(r"_", "", section)  # Remove italics
            section = re.sub(r"\n+", " ", section)  # Replace multiple newlines
            chunks.append(section)

    print(f"Generated Chunks: {chunks[:5]}")  # Print the first 5 chunks
    return chunks

# 1. Setup Embedding and Retrieval System


class BookRetriever:
    def __init__(self, book_text):
        # Initialize embedding function
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_API_KEY,
            model_name=EMBEDDING_MODEL
        )

        # Process book into chunks
        self.chunks = process_book_text(book_text)
        # Print the first 5 chunks
        print(f"Chunks to Add to ChromaDB: {self.chunks[:5]}")

        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=CHROMA_DB_NAME,
            embedding_function=self.embedding_function
        )

        # Add documents to the database
        self.collection.add(
            documents=self.chunks,
            ids=[str(i) for i in range(len(self.chunks))]
        )

    def retrieve(self, query, n_results=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        # Check if documents are retrieved
        if results and 'documents' in results and results['documents']:
            return results['documents'][0]
        else:
            return ["No relevant passages found."]

# 2. Setup Generation with Gemini


class AnswerGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel(GENERATION_MODEL)

    def generate(self, prompt):
        response = self.model.generate_content(prompt)
        return response.text

# 3. Create RAG Pipeline


class CheeseRAGSystem:
    def __init__(self, book_text):
        self.retriever = BookRetriever(book_text)
        self.generator = AnswerGenerator()

    def build_prompt(self, question, contexts):
        prompt = f"""You are an expert on the book "Who Moved My Cheese?" by Dr. Spencer Johnson.
        Answer the question using ONLY the provided context from the book. Be accurate and concise.

Question: {question}

Relevant passages from the book:
"""
        for i, context in enumerate(contexts):
            prompt += f"--- Passage {i+1} ---\n{context}\n\n"

        prompt += "\nBased on these passages, answer the question:"
        return prompt

    def ask(self, question):
        # Retrieve relevant book passages
        contexts = self.retriever.retrieve(question)

        # Build prompt
        prompt = self.build_prompt(question, contexts)

        # Generate answer
        answer = self.generator.generate(prompt)

        return answer, contexts

# 4. Create Gradio UI


def create_cheese_ui():
    def load_book(file):
        """Load the book content from the uploaded file."""
        try:
            if file is None:
                return None
            book_text = file.read().decode("utf-8")
            # Print the first 500 characters
            print(f"Loaded Book Content: {book_text[:500]}")
            return book_text
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def respond(question, history, book_file):
        if book_file is None:
            return "Please upload a book file first.", ""

        # Load the book content
        book_text = load_book(book_file)
        # Print the first 500 characters
        print(f"Book Text Loaded: {book_text[:500]}")
        if not book_text or "Error" in book_text:
            return "Error reading the uploaded file. Please try again.", ""

        # Initialize the RAG system with the uploaded book text
        rag_system = CheeseRAGSystem(book_text)

        # Generate the response
        answer, contexts = rag_system.ask(question)

        # Format the contexts for display
        context_display = "\n\n---\n\n".join(
            [f"Passage {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)]
        )

        return answer, context_display

    # Define custom CSS for styling
    custom_css = """
    body {
        background-color: #d9f2ff; /* Light blue background */
        font-family: 'Calibri', sans-serif; /* Calibri font */
    }
    .gradio-container {
        background-color: #d9f2ff; /* Light blue background for the container */
        font-family: 'Calibri', sans-serif; /* Calibri font */
    }
    .gr-textbox, .gr-button {
        font-family: 'Calibri', sans-serif; /* Ensure inputs and buttons use Calibri */
    }
    """

    with gr.Blocks(css=custom_css, title="Book QA with Gemini") as demo:
        gr.Markdown("""# 📖 GenAI Usecase - Book Question Answering
        Powered by Google Gemini""")

        with gr.Row():
            with gr.Column():
                book_file = gr.File(
                    label="Upload a Book (Text File)", file_types=[".txt"]
                )
                question_input = gr.Textbox(
                    label="Ask about the book",
                    placeholder="e.g., What are the key financial strategies?"
                )
                submit_btn = gr.Button("Get Answer")

            with gr.Column():
                answer_output = gr.Textbox(
                    label="Gemini's Answer",
                    interactive=False,
                    lines=5
                )
                contexts_output = gr.Textbox(
                    label="Relevant Book Passages",
                    interactive=False,
                    lines=10
                )

        examples = gr.Examples(
            examples=[
                "What are the income limits for Roth IRAs?",
                "Explain the benefits of a Roth 401(k).",
                "What is a Backdoor Roth IRA?",
                "What are the key financial strategies for high-income households?"
            ],
            inputs=[question_input]
        )

        # Link the button to the respond function
        submit_btn.click(
            fn=respond,
            inputs=[question_input, book_file],
            outputs=[answer_output, contexts_output]
        )

    return demo


# Run the Application
ui = create_cheese_ui()
ui.launch(server_name="0.0.0.0", server_port=7861)
