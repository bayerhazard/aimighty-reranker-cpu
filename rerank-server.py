import sys, os
import time, logging, asyncio, threading, ctypes
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Union, Optional
import uvicorn
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("reranker")

# Creative Cooperative Optimization: Run with nice value 10 (lower priority)
try:
    os.nice(10)
    log.info("Successfully set process niceness to 10 (cooperative background priority)")
except Exception as e:
    log.warning("Could not set process niceness: %s", e)

MODEL_DIR  = os.getenv("MODEL_DIR", "/models_cache/aimighty-reranker-0.6b")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
PORT       = int(os.getenv("PORT", "30001"))

OV_DEVICE = os.getenv("OV_DEVICE", "CPU")
CPU_PINNING = os.getenv("CPU_PINNING", "NO")

def _build_ov_config(device):
    cfg = {
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
    }
    if device.upper() == "CPU":
        cfg["INFERENCE_NUM_THREADS"] = os.getenv("INFERENCE_NUM_THREADS", "8")
        cfg["SCHEDULING_CORE_TYPE"] = "PCORE_ONLY"
        cfg["ENABLE_CPU_PINNING"] = "YES" if CPU_PINNING.upper() == "YES" else "NO"
    return cfg

app = FastAPI()
_model = None
_tokenizer = None
_infer_lock = threading.Lock()
_model_ready = False
try:
    _libc = ctypes.CDLL("libc.so.6", mode=ctypes.RTLD_GLOBAL)
except (AttributeError, OSError):
    _libc = None

def get_model():
    global _model, _tokenizer, _model_ready
    if _model:
        return _model, _tokenizer

    log.info("=" * 50)
    log.info("Loading model from: %s", MODEL_DIR)
    log.info("=" * 50)

    from optimum.intel import OVModelForSequenceClassification
    from transformers import AutoTokenizer

    log.info("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    log.info("Tokenizer loaded successfully.")

    log.info("Loading OpenVINO model on %s...", OV_DEVICE)

    device_to_use = OV_DEVICE
    ov_config = _build_ov_config(device_to_use)
    log.info("OpenVINO config: %s", ov_config)

    try:
        _model = OVModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            device=device_to_use,
            compile=False,
            ov_config=ov_config,
        )
        _model.compile()
        log.info("Model loaded and compiled on %s successfully.", device_to_use)
        _model_ready = True
    except Exception as e:
        log.exception("Model compilation failed: %s", e)
        raise e

    return _model, _tokenizer

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: Optional[str] = None
    top_n: Optional[int] = None
    instruction: Optional[str] = None

SYSTEM_PREFIX = "^system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".\n\n^user\n"
ASSISTANT_SUFFIX = "\n\n^assistant\n  \n\n\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

def format_pair(query, document, instruction=None):
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION
    return f"{SYSTEM_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{ASSISTANT_SUFFIX}"

def _run_inference(query: str, documents: List[str], instruction: Optional[str] = None):
    """Synchronous inference — runs in a worker thread to keep the event loop free."""
    import torch

    with _infer_lock:
        model, tok = get_model()
        log.info("Reranking %d document(s) for query: '%s'", len(documents), query[:50])

        pairs = [format_pair(query, doc, instruction) for doc in documents]
        
        # Tokenize and run model
        enc = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**enc)
        
        logits = outputs.logits
        if logits.shape[1] == 1:
            scores = torch.sigmoid(logits[:, 0]).tolist()
        else:
            # If 2 logits, we can use softmax or sigmoid of logits[:, 1]
            scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
            
        total_tokens = int(enc["input_ids"].numel())

    try:
        _libc.malloc_trim(0)
    except (AttributeError, OSError):
        pass
    return scores, total_tokens

@app.post("/rerank")
@app.post("/v1/rerank")
@app.post("/v2/rerank")
async def rerank(req: RerankRequest):
    if not req.documents:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "documents field is required", "type": "Bad Request", "param": None, "code": 400}}
        )
    if not req.query:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "query field is required", "type": "Bad Request", "param": None, "code": 400}}
        )

    try:
        scores, total_tokens = await asyncio.wait_for(
            asyncio.to_thread(_run_inference, req.query, req.documents, req.instruction),
            timeout=300.0,
        )
    except asyncio.TimeoutError:
        log.error("Reranking timeout after 300s for %d document(s)", len(req.documents))
        return JSONResponse(
            status_code=504,
            content={"error": "inference timeout", "detail": "Request exceeded 300s limit"},
        )

    results = []
    for idx, score in enumerate(scores):
        results.append({
            "index": idx,
            "document": {"text": req.documents[idx], "multi_modal": None},
            "relevance_score": float(score)
        })

    # Sort results by relevance score descending
    results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    top_n = req.top_n if req.top_n is not None else len(results)
    results = results[:top_n]

    model_name_to_return = req.model if req.model else MODEL_NAME

    return {
        "id": f"rerank-{uuid.uuid4().hex[:12]}",
        "model": model_name_to_return,
        "results": results,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

@app.get("/health")
def health():
    status = "ready" if _model_ready else "loading"
    return {"status": status}

@app.get("/v1/models")
@app.get("/get_model_info")
def models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "aimighty"
        }]
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    device = OV_DEVICE
    status_text = "Ready" if _model_ready else "Loading"
    status_class = "ready" if _model_ready else "offline"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aimighty Reranker &mdash; Qwen3-Reranker-0.6B CPU</title>
<link rel="icon" type="image/png" href="https://raw.githubusercontent.com/bayerhazard/aimighty-reranker/main/icon.png">
<style>
  *,*::before,*::after{{box-sizing:border-box}}
  html,body{{margin:0;padding:0}}
  body{{
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);
    color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center;
    padding:2rem 1rem;
  }}
  .card{{
    width:100%;max-width:860px;
    background:rgba(15,23,42,.72);
    border:1px solid rgba(148,163,184,.15);
    border-radius:20px;padding:2.5rem;
    backdrop-filter:blur(20px);
    box-shadow:0 25px 50px -12px rgba(0,0,0,.5);
  }}
  .header{{display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem}}
  .icon{{width:64px;height:64px;border-radius:14px;background:#fff;padding:6px;flex-shrink:0}}
  h1{{margin:0;font-size:1.75rem;font-weight:700;letter-spacing:-.02em}}
  .subtitle{{margin:.25rem 0 0;color:#94a3b8;font-size:.95rem}}
  .badge{{
    display:inline-flex;align-items:center;gap:.5rem;
    margin-top:.75rem;padding:.35rem .75rem;
    border-radius:999px;font-size:.8rem;font-weight:600;
  }}
  .badge.ready{{background:rgba(16,185,129,.12);border:1px solid #10b98155;color:#10b981}}
  .badge.offline{{background:rgba(239,68,68,.12);border:1px solid #ef444455;color:#ef4444}}
  .dot{{width:8px;height:8px;border-radius:50%}}
  .dot.ready{{background:#10b981;box-shadow:0 0 12px #10b981}}
  .dot.offline{{background:#ef4444;box-shadow:0 0 12px #ef4444}}
  h2{{margin:2rem 0 .75rem;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#64748b}}
  .endpoints{{display:grid;gap:.5rem}}
  .endpoint{{
    display:flex;align-items:center;gap:.75rem;
    padding:.75rem 1rem;background:rgba(30,41,59,.5);
    border:1px solid rgba(148,163,184,.1);border-radius:10px;
    font-family:"SF Mono",Monaco,Consolas,monospace;font-size:.85rem;
  }}
  .method{{
    flex-shrink:0;font-weight:700;font-size:.7rem;
    padding:.2rem .5rem;border-radius:5px;letter-spacing:.05em;
  }}
  .method.GET{{background:rgba(59,130,246,.2);color:#60a5fa}}
  .method.POST{{background:rgba(168,85,247,.2);color:#c084fc}}
  .path{{color:#e2e8f0;flex:1}}
  .desc{{color:#64748b;font-size:.8rem;font-family:-apple-system,sans-serif}}
  pre{{
    margin:0;padding:1.25rem;background:#020617;
    border:1px solid rgba(148,163,184,.1);border-radius:10px;
    font-family:"SF Mono",Monaco,Consolas,monospace;font-size:.8rem;
    color:#cbd5e1;overflow-x:auto;line-height:1.6;
  }}
  .kw{{color:#f472b6}}.str{{color:#86efac}}
  .meta{{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
    gap:.75rem;margin-top:1rem;
  }}
  .meta-item{{
    padding:.85rem 1rem;background:rgba(30,41,59,.4);
    border:1px solid rgba(148,163,184,.1);border-radius:10px;
  }}
  .meta-label{{
    font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:#64748b;font-weight:700;
  }}
  .meta-value{{
    margin-top:.3rem;color:#e2e8f0;font-size:.95rem;font-weight:600;
    font-family:"SF Mono",Monaco,Consolas,monospace;
  }}
  footer{{margin-top:2rem;padding-top:1.25rem;border-top:1px solid rgba(148,163,184,.1);color:#64748b;font-size:.8rem;text-align:center}}
  a{{color:#60a5fa;text-decoration:none}}a:hover{{color:#93c5fd}}
  @media(max-width:600px){{.card{{padding:1.5rem}}}}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <img class="icon" src="https://raw.githubusercontent.com/bayerhazard/aimighty-reranker/main/icon.png" alt="Aimighty Reranker">
    <div>
      <h1>Aimighty Reranker (CPU)</h1>
      <p class="subtitle">Qwen3-Reranker-0.6B &middot; OpenVINO &middot; Intel Core Ultra 9</p>
      <span class="badge {status_class}">
        <span class="dot {status_class}"></span>
        <span>{status_text}</span>
      </span>
    </div>
  </div>
  <h2>Model Info</h2>
  <div class="meta">
    <div class="meta-item"><div class="meta-label">Model</div><div class="meta-value">{MODEL_NAME}</div></div>
    <div class="meta-item"><div class="meta-label">Device</div><div class="meta-value">{device}</div></div>
    <div class="meta-item"><div class="meta-label">Precision</div><div class="meta-value">INT8 PTQ</div></div>
    <div class="meta-item"><div class="meta-label">Backend</div><div class="meta-value">OpenVINO 2026</div></div>
  </div>
  <h2>API Endpoints</h2>
  <div class="endpoints">
    <div class="endpoint"><span class="method POST">POST</span><span class="path">/v1/rerank</span><span class="desc">Rerank documents (Jina/Cohere-compatible)</span></div>
    <div class="endpoint"><span class="method GET">GET</span><span class="path">/v1/models</span><span class="desc">List available models</span></div>
    <div class="endpoint"><span class="method GET">GET</span><span class="path">/health</span><span class="desc">Liveness probe</span></div>
  </div>
  <h2>Quick start</h2>
  <pre><span class="kw">curl</span> -X POST <span class="str">"http://localhost:{PORT}/v1/rerank"</span> \\
  -H <span class="str">"Content-Type: application/json"</span> \\
  -d <span class="str">'{{"query": "German capital", "documents": ["Berlin is the capital.", "Paris is in France."], "top_n": 2}}'</span></pre>
  <footer>
    Built for <a href="https://github.com/bayerhazard/aimighty-reranker-cpu" target="_blank" rel="noopener">Olares</a> &middot;
    <a href="https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls" target="_blank" rel="noopener">Qwen3-Reranker-0.6B</a> &middot;
    <a href="https://docs.openvino.ai/" target="_blank" rel="noopener">OpenVINO 2026</a>
  </footer>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.on_event("startup")
def startup_event():
    log.info("FastAPI startup: warming up model...")
    try:
        model, tok = get_model()
        import torch
        
        # Warmup
        query = "warmup"
        docs = ["warmup document 1", "warmup document 2"]
        pairs = [format_pair(query, doc) for doc in docs]
        enc = tok(pairs, padding=True, truncation=True, return_tensors="pt")
        with _infer_lock:
            with torch.no_grad():
                model(**enc)
        log.info("Warmup inference complete. Model ready.")
    except Exception as e:
        log.exception("Model warmup failed: %s", e)

if __name__ == "__main__":
    log.info("Starting uvicorn on port %d", PORT)
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, workers=1)
