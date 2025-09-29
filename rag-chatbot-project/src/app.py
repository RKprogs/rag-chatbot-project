import gradio as gr
from .retriever import Retriever
from .llm_wrapper import LLMWrapper
from .function_caller import wiki_summary
import os, threading, time

# Configurable params
INDEX_PATH = os.environ.get('INDEX_PATH', 'data/index.faiss')
META_PATH = os.environ.get('META_PATH', 'data/index_meta.json')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
LLM_MODEL = os.environ.get('LLM_MODEL', 'google/flan-t5-small')

retriever = Retriever(index_path=INDEX_PATH, meta_path=META_PATH, model_name=EMBED_MODEL)
llm = LLMWrapper(model_name=LLM_MODEL, max_length=256)

def generate_answer(query: str, use_wiki: bool = False, top_k: int = 4):
    # optional wiki augmentation
    wiki_text = ''
    if use_wiki:
        # if user inputs "wiki:topic" handle that
        if query.lower().startswith('wiki:'):
            topic = query.split(':',1)[1].strip()
        else:
            topic = query
        w = wiki_summary(topic)
        if w:
            wiki_text = f"Wikipedia summary for {topic}:\n{w}\n\n"
    docs = retriever.retrieve(query, top_k=top_k)
    context = "\n---\n".join([d['text'] for d in docs])
    prompt = f"{wiki_text}Use the following extracted document snippets to answer the question concisely.\n\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = llm.answer(prompt)
    return answer

def run_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Chatbot — Colab Demo\nAsk domain-specific questions based on ingested PDFs.")
        with gr.Row():
            query = gr.Textbox(lines=2, placeholder='Enter your question here...')
            wiki_checkbox = gr.Checkbox(label='Augment with Wikipedia summary')
        output = gr.Textbox(label='Answer', lines=6)
        def _chat(q, w):
            return generate_answer(q, use_wiki=w)
        query.submit(_chat, inputs=[query, wiki_checkbox], outputs=output)
        btn = gr.Button('Ask')
        btn.click(_chat, inputs=[query, wiki_checkbox], outputs=output)
    demo.launch(share=True)

if __name__ == '__main__':
    # Running inside Colab: launching in a thread avoids blocking cell
    t = threading.Thread(target=run_gradio, daemon=True)
    t.start()
    # keep main thread alive while Gradio runs
    while True:
        time.sleep(1)
