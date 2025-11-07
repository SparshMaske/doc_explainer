# backend/explain.py
import os


# Choose one: OpenAI or local HF model. Here we show a simple OpenAI example and a safe fallback summary.


OPENAI_KEY = os.environ.get("OPENAI_API_KEY")


if OPENAI_KEY:
from openai import OpenAI
client = OpenAI(api_key=OPENAI_KEY)


def explain_text(text: str) -> str:
prompt = (
"You are an insurance assistant. Read the following claim or settlement text and write a concise, plain-language explanation (max 200 words).\n\n" + text
)
try:
resp = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=300)
return resp.output[0].content[0].text
except Exception as e:
return f"(explain_error) {str(e)}"
else:
# Simple fallback summarizer: naive rule-based summary
def explain_text(text: str) -> str:
# pick first 2-3 paragraphs and simplify
paras = [p.strip() for p in text.split("\n\n") if p.strip()]
if not paras:
return "No explanation available (empty document)."
summary = " ".join(paras[:3])
if len(summary) > 1000:
summary = summary[:1000] + "..."
return summary
