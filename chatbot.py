import logging
import gradio as gr
from agent import build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    agent = build_agent()

    def chat(message, history):
        try:
            result = agent.invoke({"question": message, "history": history},
                                  config={"recursion_limit": 50})
            logger.info("Agent trace: %s", result.get("trace"))
            return result["answer"]
        except Exception:
            logger.exception("Chat request failed")
            return "The request failed. Check the application logs and API configuration."

    demo = gr.ChatInterface(
        fn=chat, title="Aviation Accident Self-RAG Assistant",
        description="Research prototype. Not operational aviation advice. Uses supplied reports; "
                    "PDF page numbers may differ from printed page labels. Text is sent to OpenAI.",
        examples=["What was the probable cause of the VT-PTE accident?",
                  "Compare contributing factors for VT-PTE and VT-GDI.",
                  "What findings are preliminary in the VT-ANB report?"])
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
