"""
Simplified Gradio frontend for debugging.
"""

import asyncio
import gradio as gr

def simple_echo(question: str, top_k: int) -> str:
    """Simple echo function for testing."""
    return f"Echo: {question} (top_k: {top_k})"

def test_search_function(question: str, top_k: int) -> str:
    """Test the actual search function."""
    try:
        # Import inside function to avoid startup delays
        import sys
        sys.path.append('.')
        from app.retrieval import search_and_answer
        
        # Test async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_search():
            response = await search_and_answer(question, top_k)
            return f"""## Answer
{response.answer}

## Source
**Title**: {response.source.title if response.source else 'None'}
**URL**: {response.source.url if response.source else 'None'}
**Relevance**: {response.source.relevance if response.source else 'N/A'}
"""
        
        result = loop.run_until_complete(run_search())
        loop.close()
        return result
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create interface
with gr.Blocks(title="Blog Q&A Test") as demo:
    gr.Markdown("# 🤖 Blog Q&A Chatbot (Test)")
    
    question = gr.Textbox(label="Question", placeholder="Ask something...")
    top_k = gr.Slider(1, 10, 5, label="Results Count")
    submit = gr.Button("Test Search")
    
    output = gr.Markdown(value="Ready to test...")
    
    submit.click(test_search_function, [question, top_k], output)

if __name__ == "__main__":
    print("🚀 Starting simplified Gradio test...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=8002,
        share=False,
        show_error=True
    )