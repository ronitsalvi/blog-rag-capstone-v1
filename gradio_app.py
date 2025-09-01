"""
Gradio frontend for the Blog Q&A Chatbot system.
"""

import asyncio
import gradio as gr
from app.retrieval import search_and_answer
from app.models import QuestionResponse

def format_response(response: QuestionResponse) -> str:
    """Format the chatbot response for display in Gradio."""
    
    if not response:
        return "❌ **Error**: No response received from the system."
    
    # Start building the formatted response
    output = []
    
    # Main answer
    if response.answer:
        output.append(f"## 💡 Answer\n\n{response.answer}")
    
    # Source information
    if response.source:
        output.append(f"\n## 📚 Source")
        output.append(f"**Title**: {response.source.title}")
        output.append(f"**URL**: [{response.source.url}]({response.source.url})")
        output.append(f"**Relevance**: {response.source.relevance:.2f}")
    
    # Excerpt if available
    if response.excerpt:
        output.append(f"\n## 📝 Excerpt\n\n> {response.excerpt}")
    
    # Additional citations
    if response.citations and len(response.citations) > 0:
        output.append(f"\n## 🔗 Additional Sources")
        for i, citation in enumerate(response.citations, 1):
            output.append(f"{i}. [{citation.title}]({citation.url})")
    
    # Fallback indicator
    if response.fallback_used:
        output.append(f"\n## ⚠️ Fallback Used")
        output.append("This response used web search fallback.")
    
    # Policy reason if any
    if response.policy_reason:
        output.append(f"\n## ℹ️ Policy Note")
        output.append(response.policy_reason)
    
    return "\n".join(output)

async def ask_question_async(question: str, top_k: int) -> str:
    """
    Async wrapper for the search_and_answer function.
    """
    try:
        if not question or not question.strip():
            return "❌ **Error**: Please enter a question."
        
        response = await search_and_answer(question.strip(), top_k)
        return format_response(response)
        
    except Exception as e:
        return f"❌ **Error**: {str(e)}\n\nPlease try again or check that the vector index is built."

def ask_question_sync(question: str, top_k: int) -> str:
    """
    Synchronous wrapper that runs the async function.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(ask_question_async(question, top_k))
        loop.close()
        return result
    except Exception as e:
        return f"❌ **Error**: {str(e)}"

# Sample questions for user guidance
SAMPLE_QUESTIONS = [
    "What is machine learning?",
    "How do I use pandas for data analysis?",
    "What are the benefits of deep learning?",
    "How to create data visualizations?",
    "What is the difference between supervised and unsupervised learning?",
    "How to handle missing data in datasets?",
    "What is feature engineering?",
    "How to use matplotlib for plotting?"
]

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: auto !important;
}
.output-markdown {
    font-size: 14px !important;
    line-height: 1.6 !important;
}
.input-text {
    font-size: 16px !important;
}
"""

def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="Blog Q&A Chatbot",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        # Header
        gr.Markdown("""
        # 🤖 Blog Q&A Chatbot
        
        Ask questions about **data science**, **machine learning**, **Python**, and **analytics**.  
        Get answers from 955+ curated blog articles with source citations.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about data science, machine learning, Python, pandas, etc...",
                    lines=2,
                    max_lines=4
                )
                
                # Top-k slider
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Results",
                    info="How many blog sources to search through"
                )
                
                # Buttons
                with gr.Row():
                    submit_btn = gr.Button("🔍 Ask Question", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            with gr.Column(scale=2):
                # Sample questions
                gr.Markdown("### 💡 Try These Questions:")
                sample_buttons = []
                for question in SAMPLE_QUESTIONS[:4]:
                    btn = gr.Button(question, size="sm")
                    sample_buttons.append(btn)
        
        # Output area
        output_display = gr.Markdown(
            label="Response",
            value="👋 **Welcome!** Ask a question above to get started.",
            line_breaks=True
        )
        
        # Event handlers
        def clear_interface():
            return "", 5, "👋 **Welcome!** Ask a question above to get started."
        
        def set_sample_question(question):
            return question
        
        # Submit on button click or Enter
        submit_btn.click(
            fn=ask_question_sync,
            inputs=[question_input, top_k_slider],
            outputs=output_display,
            show_progress=True
        )
        
        question_input.submit(
            fn=ask_question_sync,
            inputs=[question_input, top_k_slider],
            outputs=output_display,
            show_progress=True
        )
        
        # Clear button
        clear_btn.click(
            fn=clear_interface,
            outputs=[question_input, top_k_slider, output_display]
        )
        
        # Sample question buttons
        for btn, question in zip(sample_buttons, SAMPLE_QUESTIONS[:4]):
            btn.click(
                fn=set_sample_question,
                inputs=gr.State(question),
                outputs=question_input
            )
        
        # Footer
        gr.Markdown("""
        ---
        **Data Source**: 955+ blog articles from 360DigiTMG  
        **Search Method**: Vector similarity using sentence-transformers  
        **Updated**: Local blog content with real-time search
        """)
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Blog Q&A Chatbot Interface...")
    print("📊 Loading vector index and models...")
    
    # Create and launch interface
    interface = create_interface()
    
    print("✅ Ready! Launching Gradio interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )