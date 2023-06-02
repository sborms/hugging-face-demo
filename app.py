from transformers import pipeline
import gradio as gr

model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

input = gr.Textbox(label="Prompt", placeholder="Enter text to summarize...", lines=10)
output = gr.Textbox(label="Summary")

intf = gr.Interface(fn=predict, inputs=input, outputs=output)

intf.launch()