"""
app.py
------
Gradio-based demo UI for the Meridian Capital Group HR Policy RAG pipeline.

Provides a clean chat interface where employees can ask questions about
company HR policies and receive grounded answers with source citations.

Features:
    - Chat interface with conversation history
    - Source citations displayed alongside each answer
    - Before/After toggle to compare answers with and without RAG
    - Example questions to help users get started

Usage:
    gradio ui/app.py

    # Or run directly
    python ui/app.py
"""

import logging
import os
import sys

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Initialize the pipeline once at startup
# Gradio runs this module-level code once when the app launches.
# All user requests share the same pipeline instance.
# -------------------------------------------------------------------
try:
    from src.pipeline import RAGPipeline
    logger.info("Initializing RAG pipeline for Gradio app...")
    pipeline = RAGPipeline()
    logger.info("Pipeline ready")
    PIPELINE_READY = True
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    PIPELINE_READY = False
    INIT_ERROR = str(e)


# -------------------------------------------------------------------
# Example questions — shown in the UI to guide users
# -------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "How many PTO days do I get in my first year?",
    "What is the 401(k) company match?",
    "How does the parental leave policy work?",
    "What are the steps in the disciplinary process?",
    "What is the target bonus for a Vice President?",
    "How do RSUs vest for eligible employees?",
    "What happens if I receive a rating of 2 in my annual review?",
    "How do I report harassment or discrimination anonymously?",
]


def format_sources(sources: list[dict]) -> str:
    """
    Format source citations for display in the Gradio UI.

    Args:
        sources (list[dict]): Source dicts from pipeline response.

    Returns:
        str: Markdown-formatted source citations.
    """
    if not sources:
        return "_No sources retrieved._"

    lines = ["**Sources:**\n"]
    for i, src in enumerate(sources, 1):
        lines.append(
            f"{i}. `{src['file_name']}` — "
            f"Page {src['page_label']} "
            f"_(relevance: {src['similarity_score']})_"
        )
    return "\n".join(lines)


def chat(
    message: str,
    history: list,
    show_sources: bool,
) -> tuple[list, str]:

    if not message or not message.strip():
        return history, ""

    if not PIPELINE_READY:
        error_msg = f"Pipeline failed to initialize: {INIT_ERROR}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""

    try:
        response = pipeline.query(message)
        answer = response["answer"]
        sources_md = format_sources(response["sources"]) if show_sources else ""

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, sources_md

    except Exception as e:
        logger.error(f"Query error: {e}")
        error_msg = "An error occurred while processing your question. Please try again."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


def compare_with_without_rag(question: str) -> tuple[str, str]:
    """
    Run the same question with and without RAG for comparison.

    Without RAG: calls the LLM directly with no retrieved context.
    With RAG: runs the full pipeline with retrieval.

    This is the key demo feature — shows the value of RAG clearly.

    Args:
        question (str): The user's question.

    Returns:
        Tuple of (without_rag_answer, with_rag_answer).
    """
    if not question or not question.strip():
        return "", ""

    if not PIPELINE_READY:
        return f"Pipeline error: {INIT_ERROR}", ""

    try:
        from src.generation.generator import get_llm, generate

        # Without RAG — direct LLM call with no context
        llm = get_llm()
        no_rag_prompt = (
            f"Answer this question about Meridian Capital Group HR policies:\n\n"
            f"{question}\n\n"
            f"Answer:"
        )
        no_rag_response = llm.complete(no_rag_prompt)
        without_rag = no_rag_response.text.strip()

        # With RAG — full pipeline
        response = pipeline.query(question)
        with_rag = response["answer"]

        return without_rag, with_rag

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        return f"Error: {e}", ""


# -------------------------------------------------------------------
# Build the Gradio UI
# -------------------------------------------------------------------
with gr.Blocks(
    title="Meridian Capital Group — HR Assistant",
) as demo:

    gr.Markdown(
        """
        # 🏦 Meridian Capital Group — HR Policy Assistant
        Ask questions about company policies, benefits, PTO, performance reviews, and more.
        Answers are grounded in the official Employee Handbook.
        """
    )

    with gr.Tabs():

        # -----------------------------------------------------------
        # Tab 1: Chat Interface
        # -----------------------------------------------------------
        with gr.Tab("💬 Ask HR"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="HR Assistant",
                        height=450,
                        show_label=True,
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a question about HR policies...",
                            label="Your question",
                            scale=4,
                            lines=1,
                        )
                        submit_btn = gr.Button("Ask", variant="primary", scale=1)

                    show_sources = gr.Checkbox(
                        label="Show source citations",
                        value=True,
                    )
                    clear_btn = gr.Button("Clear chat", variant="secondary")

                with gr.Column(scale=1):
                    sources_display = gr.Markdown(
                        label="Sources",
                        value="",
                )

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=msg_input,
                label="Example questions",
            )

            # Wire up interactions
            submit_btn.click(
                fn=chat,
                inputs=[msg_input, chatbot, show_sources],
                outputs=[chatbot, sources_display],
            ).then(
                fn=lambda: "",
                outputs=msg_input,
            )

            msg_input.submit(
                fn=chat,
                inputs=[msg_input, chatbot, show_sources],
                outputs=[chatbot, sources_display],
            ).then(
                fn=lambda: "",
                outputs=msg_input,
            )

            clear_btn.click(
                fn=lambda: ([], "_Sources will appear here after your first question._"),
                outputs=[chatbot, sources_display],
            )

        # -----------------------------------------------------------
        # Tab 2: Before / After RAG Comparison
        # -----------------------------------------------------------
        with gr.Tab("⚖️ RAG vs No RAG"):
            gr.Markdown(
                """
                ### See the difference RAG makes
                Ask the same question with and without retrieval.
                The model has no knowledge of Meridian Capital Group's specific policies —
                RAG is what makes accurate answers possible.
                """
            )

            compare_input = gr.Textbox(
                placeholder="Enter a question to compare...",
                label="Question",
                lines=2,
            )
            compare_btn = gr.Button("Compare", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ❌ Without RAG")
                    no_rag_output = gr.Textbox(
                        label="LLM answer (no retrieval)",
                        lines=8,
                        interactive=False,
                    )
                with gr.Column():
                    gr.Markdown("### ✅ With RAG")
                    with_rag_output = gr.Textbox(
                        label="RAG pipeline answer",
                        lines=8,
                        interactive=False,
                    )

            gr.Examples(
                examples=[
                    "How many PTO days do I get in my first year?",
                    "What is the 401(k) company match percentage?",
                    "What is the target bonus for a Vice President?",
                ],
                inputs=compare_input,
                label="Try these examples",
            )

            compare_btn.click(
                fn=compare_with_without_rag,
                inputs=compare_input,
                outputs=[no_rag_output, with_rag_output],
            )

        # -----------------------------------------------------------
        # Tab 3: About
        # -----------------------------------------------------------
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About This Project

                This demo is a **Retrieval-Augmented Generation (RAG)** pipeline built
                over a fictional HR Employee Handbook for Meridian Capital Group,
                a mid-size financial services firm.

                ### How It Works
                1. **Ingest** — the employee handbook PDF is loaded and split into chunks
                2. **Embed** — each chunk is converted to a vector using OpenAI embeddings
                3. **Index** — vectors are stored in a Chroma vector database
                4. **Retrieve** — your question is embedded and the most relevant chunks are fetched
                5. **Generate** — the LLM answers your question using only the retrieved context

                ### Tech Stack
                - **LLM**: OpenAI GPT-4o
                - **Embeddings**: OpenAI text-embedding-3-small
                - **Vector Store**: Chroma (local)
                - **Framework**: LlamaIndex
                - **UI**: Gradio

                ### Production Considerations
                In a production system this would be backed by a FastAPI service,
                a React frontend with streaming, Pinecone for scalable vector storage,
                Redis for session memory, and LangSmith for observability.

                ### Source Code
                [github.com/tohio/rag-pipeline](https://github.com/tohio/rag-pipeline)
                """
            )

# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
)
