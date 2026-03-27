"""
app.py — Interfaccia Web per Medical VQA (Streamlit)
=====================================================
Questo file implementa l'applicazione web interattiva che permette a un utente
di caricare un'immagine radiologica, digitare una domanda clinica e ricevere
una risposta generata dal modello VQA addestrato su VQA-RAD.

Flusso dell'applicazione:
  1. L'utente carica un'immagine e scrive una domanda tramite l'interfaccia Streamlit
  2. Il tipo di domanda (CLOSED / OPEN) e la categoria clinica vengono inferiti automaticamente
  3. L'immagine e la domanda contestualizzata vengono pre-elaborate e passate al modello
  4. Il modello genera una risposta tramite beam search e calcola un punteggio di confidenza
  5. Se la risposta non è vuota, Gemini 2.5 Flash produce una spiegazione clinica dettagliata
  6. L'interfaccia mostra risposta, confidenza, metadati e spiegazione in un layout a due colonne

Componenti principali:
  - CustomMedVQAModel: architettura ViT-Large + Bio_ClinicalBERT + Cross-Attention + Dual Head
  - infer_question_type / infer_answer_type: logica rule-based per classificare la domanda
  - preprocess: prepara immagine e testo per il modello
  - gemini_explain: chiama Gemini per generare la spiegazione clinica con retry automatico
  - UI Streamlit: layout a due colonne con tema dark personalizzato via CSS inline
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel, AutoTokenizer, AutoImageProcessor
from transformers import logging as hf_logging
from PIL import Image
from google import genai
from google.genai import types
import re
import time
import warnings

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
#
# Variabili globali che configurano il comportamento dell'app:
# - GEMINI_API_KEY / GEMINI_MODEL: credenziali e modello per le spiegazioni cliniche
# - MODEL_PATH: percorso locale del checkpoint .pth del modello addestrato
# - DEVICE: GPU se disponibile, altrimenti CPU
# - CONFIDENCE_THRESHOLD_*: soglie sotto le quali la risposta viene segnalata come incerta
# - QTYPE_KEYWORDS: dizionario per inferire la categoria della domanda (posizione, dimensione, ecc.)
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = "AIzaSyB5w82vpS7dBOEy-DAicBwObG7_9KgSqiE"          
GEMINI_MODEL     = "gemini-2.5-flash"
MODEL_PATH       = "C:/Users/angel/OneDrive/Desktop/ProgettoNLP/models/best_vqa_model.pth"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD_CLOSED = 0.80
CONFIDENCE_THRESHOLD_OPEN   = 0.50

# Keyword per inferire il tipo di domanda dalla domanda testuale
QTYPE_KEYWORDS = {
    "PRES":  ["present", "there", "is there", "are there", "visible", "seen", "show"],
    "POS":   ["where", "location", "located", "position", "side", "region"],
    "PLANE": ["plane", "view", "orientation", "axial", "sagittal", "coronal"],
    "ABN":   ["abnormal", "abnormality", "finding", "pathology", "disease", "lesion"],
    "SIZE":  ["size", "large", "small", "enlarg", "dimension", "measure"],
    "ATTR":  ["type", "appearance", "density", "signal", "intensity", "shape"],
    "COUNT": ["how many", "number", "count", "multiple"],
}

# ─────────────────────────────────────────────────────────────────────────────
# MODELLO
#
# Replica esatta dell'architettura usata durante il training (medicalvqa.ipynb).
# È fondamentale che questa classe sia identica a quella del notebook: qualsiasi
# differenza nella struttura dei layer causerebbe un errore nel load_state_dict.
#
# Architettura:
#   - ViT-Large-patch32-384: encoder visivo, congelato tranne gli ultimi 3 layer
#   - Bio_ClinicalBERT embeddings: layer di embedding per le domande
#   - Cross-Attention (8 teste): fonde feature visive e linguistiche
#   - Closed Head (MLP): per domande sì/no
#   - Open Head (Transformer Decoder + beam search): per domande descrittive
#
# Il metodo generate() calcola anche uno score di confidenza:
#   - Per CLOSED: probabilità massima del softmax della closed head
#   - Per OPEN: best beam score normalizzato per lunghezza, convertito in probabilità lineare
# ─────────────────────────────────────────────────────────────────────────────
class CustomMedVQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=1024):
        super().__init__()
        
        self.vision_encoder = ViTModel.from_pretrained("google/vit-large-patch32-384")
        
        # --- TECNICA: FREEZING ---
        # Blocchiamo i gradienti per l'encoder visivo per preservare i pesi pre-addestrati
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        n_layers_to_unfreeze = 3
        for layer in self.vision_encoder.encoder.layer[-n_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        for param in self.vision_encoder.layernorm.parameters():
            param.requires_grad = True
        
        # 2. Embedding Linguistici (Bio_ClinicalBERT)
        bert_base = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.embedding = bert_base.embeddings.word_embeddings 
        self.vision_projection = nn.Linear(1024, 768)
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(768)

        # --- HEAD 1: Domande Chiuse (MLP) ---
        self.closed_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, vocab_size) 
        )

        # --- HEAD 2: Domande Aperte (Decoder) ---
        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True, dropout=0.27150897038028765)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        self.fc_out = nn.Linear(768, vocab_size)
        self.fc_out.weight = self.embedding.weight


    def forward(self, pixel_values, question_ids, answer_ids):
        # Estrazione feature e Cross-Attention Fusion
        vision_feats = self.vision_encoder(pixel_values).last_hidden_state
        vision_feats = self.vision_projection(vision_feats)
        question_feats = self.embedding(question_ids)
        
        attn_output, _ = self.cross_attention(query=vision_feats, key=question_feats, value=question_feats)
        memory = self.layer_norm(vision_feats + attn_output) 
        
        # Output Head 1 (Domande Chiuse)
        pooled_memory = memory.mean(dim=1)
        closed_logits = self.closed_head(pooled_memory)

        # Output Head 2 (Domande Aperte)
        answer_embeds = self.embedding(answer_ids)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(answer_ids.size(1)).to(pixel_values.device)
        output = self.decoder(tgt=answer_embeds, memory=memory, tgt_mask=tgt_mask)
        open_logits = self.fc_out(output)
        
        return closed_logits, open_logits

    @torch.no_grad()
    def generate(self, pixel_values, question_ids, question_types,
                 tokenizer, num_beams=3, max_len=32):
        self.eval()
        batch_size = pixel_values.size(0)
        dev        = pixel_values.device
 
        vision_feats   = self.vision_encoder(pixel_values).last_hidden_state
        vision_feats   = self.vision_projection(vision_feats)
        question_feats = self.embedding(question_ids)
        attn_output, _ = self.cross_attention(query=vision_feats,
                                               key=question_feats,
                                               value=question_feats)
        memory = self.layer_norm(vision_feats + attn_output)
 
        pooled_memory = memory.mean(dim=1)
        closed_logits = self.closed_head(pooled_memory)
        closed_preds  = closed_logits.argmax(dim=-1)
 
        # ── NUOVO: confidence CLOSED = prob max dopo softmax ────────────────
        closed_probs       = torch.softmax(closed_logits, dim=-1)
        closed_confidence  = closed_probs.max(dim=-1).values  # (batch_size,)
 
        memory_expanded = memory.repeat_interleave(num_beams, dim=0)
        generated       = torch.full((batch_size * num_beams, 1),
                                     tokenizer.cls_token_id,
                                     dtype=torch.long).to(dev)
        beam_scores = torch.zeros((batch_size, num_beams)).to(dev)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
 
        for _ in range(max_len):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1)
            ).to(dev)
            output            = self.decoder(tgt=self.embedding(generated),
                                             memory=memory_expanded,
                                             tgt_mask=tgt_mask)
            next_token_logits = self.fc_out(output[:, -1, :])
            next_token_probs  = torch.log_softmax(next_token_logits, dim=-1)
 
            next_scores = next_token_probs + beam_scores[:, None]
            next_scores = next_scores.view(batch_size,
                                           num_beams * next_token_probs.size(-1))
 
            topk_scores, topk_indices = torch.topk(next_scores, num_beams, dim=1)
            beam_ids  = topk_indices // next_token_probs.size(-1)
            token_ids = topk_indices %  next_token_probs.size(-1)
 
            new_generated = []
            for i in range(batch_size):
                for j in range(num_beams):
                    prev_idx = i * num_beams + beam_ids[i, j]
                    new_seq  = torch.cat([generated[prev_idx],
                                          token_ids[i, j].unsqueeze(0)])
                    new_generated.append(new_seq)
 
            generated   = torch.stack(new_generated)
            beam_scores = topk_scores.view(-1)
            if (token_ids == tokenizer.sep_token_id).all():
                break
 
        best_generated = generated.view(batch_size, num_beams, -1)[:, 0, :]
 
        # ── NUOVO: confidence OPEN = best beam score normalizzato ───────────
        # best_beam_score è una log-prob cumulativa → la normalizziamo
        # per lunghezza e la convertiamo in probabilità lineare
        best_beam_scores = beam_scores.view(batch_size, num_beams)[:, 0]
        seq_len          = best_generated.size(1)
        open_confidence  = torch.exp(best_beam_scores / max(seq_len, 1))
        # clamp in [0,1] per sicurezza numerica
        open_confidence  = open_confidence.clamp(0.0, 1.0)
 
        # ── Routing finale (identico all'originale) ──────────────────────────
        confidences = torch.zeros(batch_size, device=dev)
        for i in range(batch_size):
            if question_types[i] == 0:   # CLOSED
                best_generated[i]    = tokenizer.pad_token_id
                best_generated[i, 0] = tokenizer.cls_token_id
                best_generated[i, 1] = closed_preds[i]
                best_generated[i, 2] = tokenizer.sep_token_id
                confidences[i]       = closed_confidence[i]
            else:                        # OPEN
                confidences[i]       = open_confidence[i]
 
        # ──  restituisce anche confidences ──────────────────────────
        return best_generated, confidences   




# ─────────────────────────────────────────────────────────────────────────────
# FUNZIONI DI SUPPORTO
#
# infer_question_type(question):
#   Regola rule-based che cerca keyword nella domanda per determinare la categoria
#   clinica (PRES=presenza, POS=posizione, ABN=anomalia, SIZE=dimensione, ecc.).
#   Usata per costruire il prefisso contestuale [ORGANO | TIPO].
#
# infer_answer_type(question):
#   Determina se la domanda è CLOSED (sì/no) o OPEN (risposta libera) guardando
#   il verbo iniziale con espressioni regolari. Le domande che iniziano con is/are/
#   was/were/does/do/has/have/can/could/did sono classificate come CLOSED.
#
# load_model_and_tokenizer():
#   Carica il tokenizer, l'image processor e il modello dal checkpoint su disco.
#   Decorata con @st.cache_resource per evitare ricaricamenti ad ogni interazione.
#
# preprocess(image, question, organ, q_type, ...):
#   Pre-elabora l'immagine (converti in RGB, applica image processor) e la domanda
#   (costruisci la versione contestualizzata, tokenizza). Restituisce i tensori
#   pronti per essere passati al modello.
#
# gemini_explain(question, answer, organ, q_type, ans_type, confidence):
#   Costruisce un prompt strutturato per Gemini con tutti i metadati della predizione
#   e chiama l'API. Implementa retry con backoff esponenziale per il rate limit (429).
#   Aggiunge una nota sulla confidenza nel prompt per guidare il tono della spiegazione.
# ─────────────────────────────────────────────────────────────────────────────
def infer_question_type(question: str) -> str:
    q = question.lower()
    for qtype, keywords in QTYPE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return qtype
    return "UNKNOWN"

def infer_answer_type(question: str) -> tuple[str, int]:
    """Ritorna (ans_type_str, question_type_idx)."""
    q = question.lower().strip()
    closed_patterns = [
        r"^is\b", r"^are\b", r"^was\b", r"^were\b",
        r"^does\b", r"^do\b", r"^has\b", r"^have\b",
        r"^can\b", r"^could\b", r"^did\b",
    ]
    if any(re.match(p, q) for p in closed_patterns):
        return "CLOSED", 0
    return "OPEN", 1

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    tokenizer       = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-large-patch32-384", use_fast=True)
    vocab_size      = len(tokenizer)
    model           = CustomMedVQAModel(vocab_size=vocab_size).to(DEVICE)
    state_dict      = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ MODELLO CUSTOM CARICATO DA DISCO: {MODEL_PATH}")
    return model, tokenizer, image_processor

def preprocess(image: Image.Image, question: str, organ: str, q_type: str,
               tokenizer, image_processor, max_len: int = 512):
    image = image.convert("RGB")
    contextualized_q = f"[{organ} | {q_type}] {question}"
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    tokens       = tokenizer(contextualized_q, truncation=True,
                              padding="max_length", max_length=max_len,
                              return_tensors="pt")
    input_ids      = tokens.input_ids.to(DEVICE)
    attention_mask = tokens.attention_mask.to(DEVICE)
    return pixel_values, input_ids, attention_mask

def gemini_explain(question: str, answer: str, organ: str, q_type: str,
                   ans_type: str, confidence: float) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    confidence_note = (
        "The model answered with high confidence."
        if confidence >= (CONFIDENCE_THRESHOLD_CLOSED if ans_type == "CLOSED"
                          else CONFIDENCE_THRESHOLD_OPEN)
        else f"Note: the model answered with relatively low confidence ({confidence:.1%}), "
             "so treat this answer with extra caution."
    )

    prompt = f"""You are a radiology expert assistant. A medical VQA model has analyzed 
a radiology image and produced the following answer. Your job is to explain 
the answer clearly and concisely to a medical professional.

Anatomical region: {organ}
Question category: {q_type}
Question type: {ans_type}
Question: {question}
Model answer: {answer}
Model confidence: {confidence:.1%}
{confidence_note}

Instructions:
- Explain what the answer means clinically in 2-4 sentences
- Mention any relevant clinical implications if appropriate
- If confidence is low, suggest verifying with a specialist
- Use clear, professional medical language
- Do NOT repeat the question or answer verbatim at the start

Explanation:"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=600,
                    temperature=0.3,
                )
            )
            text = response.text.strip()
            if text and text[-1] not in ".!?":
                text += "..."
            return text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(4 * (attempt + 1))
            else:
                return f"⚠️ Gemini non disponibile: {e}"
    return "⚠️ Gemini non disponibile dopo 3 tentativi."


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
#
# Layout a due colonne:
#   - Colonna sinistra (col_left): upload immagine, input domanda, scelta organo, bottone
#   - Colonna destra (col_right): risultati (risposta, metriche, spiegazione Gemini)
#
# Flusso all'interazione:
#   1. Alla prima esecuzione, mostra una progress bar mentre carica il modello
#      (il caricamento successivo è istantaneo grazie a st.cache_resource)
#   2. Inferisce organo, q_type e ans_type dalla domanda e dalla scelta dell'utente
#   3. Pre-elabora e chiama model.generate() per ottenere risposta e confidenza
#   4. Calcola il colore del badge di confidenza (verde/giallo/rosso) in base alla soglia
#   5. Se la risposta non è vuota e la confidenza è accettabile, chiama gemini_explain()
#
# Stile: tema dark personalizzato via CSS inline iniettato con st.markdown(unsafe_allow_html).
#   Font DM Serif Display (titoli), DM Mono (label), DM Sans (testo).
#   Variabili CSS per colori, bordi e accent — facilmente modificabili.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedVQA · Radiology Assistant",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #58a6ff;
    --accent2:   #3fb950;
    --warn:      #d29922;
    --danger:    #f85149;
    --text:      #e6edf3;
    --muted:     #8b949e;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1100px; }

h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem !important;
    letter-spacing: -0.5px;
    color: var(--text) !important;
    margin-bottom: 0 !important;
}

.subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Text input */
[data-testid="stTextInput"] > div > div > input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #0d1117 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Answer card */
.answer-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-left: 4px solid var(--accent2);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin: 1.5rem 0;
}
.answer-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.4rem;
}
.answer-text {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--text);
    line-height: 1.2;
}

/* Metric cards */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1;
    min-width: 130px;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 500;
    color: var(--text);
}
.metric-value.good  { color: var(--accent2); }
.metric-value.warn  { color: var(--warn); }
.metric-value.bad   { color: var(--danger); }

/* Confidence bar */
.conf-bar-bg {
    background: var(--border);
    border-radius: 999px;
    height: 6px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* Gemini explanation */
.gemini-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-top: 1rem;
}
.gemini-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}
.gemini-text {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--text);
}

/* Divider */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* Spinner override */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>MedVQA</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Radiology Visual Question Answering · BERTScore F1 85%</p>',
            unsafe_allow_html=True)

# ── Layout principale ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("#### 🩻 Immagine radiologica")
    uploaded = st.file_uploader("Carica immagine radiologica", type=["png", "jpg", "jpeg"],
                                  label_visibility="collapsed")
    image = None
    if uploaded:
        uploaded.seek(0)
        image = Image.open(uploaded).copy()
        st.image(image, width='stretch')


    st.markdown("#### 💬 Domanda clinica")
    question = st.text_input("Domanda clinica", placeholder="Es: Is there a fracture visible?",
                              label_visibility="collapsed")

    st.markdown("#### 🫀 Organo anatomico")
    organ_choice = st.selectbox(
        "Organo anatomico",
        options=["HEAD", "CHEST", "ABD"],
        format_func=lambda x: {
            "HEAD":  "🧠 Head / Brain (CT · MRI)",
            "CHEST": "🫁 Chest / Lung / Heart (X-Ray)",
            "ABD":   "🫃 Abdomen (CT)",
        }[x],
        label_visibility="collapsed"
    )
    run = st.button("Analizza immagine →")

with col_right:
    if run:
        if not uploaded:
            st.warning("⚠️ Carica prima un'immagine radiologica.")
        elif not question.strip():
            st.warning("⚠️ Inserisci una domanda clinica.")
        else:
            # ── Verifica che l'immagine sia stata caricata correttamente ──
            if image is None:
                st.warning("⚠️ Errore nel caricamento dell'immagine. Ricaricala.")
                st.stop()

            # ── Caricamento modello con progress bar ───────────────────────
            is_first_load = "model_loaded" not in st.session_state
            if is_first_load:
                bar = st.progress(0, text="⏳ Inizializzazione modello...")
                bar.progress(10, text="⏳ Caricamento tokenizer Bio_ClinicalBERT...")
                model, tokenizer, image_processor = load_model_and_tokenizer()
                bar.progress(90, text="⏳ Finalizzazione...")
                bar.progress(100, text="✅ Modello pronto!")
                time.sleep(0.6)
                bar.empty()
                st.session_state["model_loaded"] = True
            else:
                model, tokenizer, image_processor = load_model_and_tokenizer()

            # Organo: sempre dalla scelta dell'utente
            organ   = organ_choice
            q_type  = infer_question_type(question)
            ans_type_str, q_type_idx = infer_answer_type(question)

            
            
            print(f"🔍 INFERENZA: domanda='{question}', organo='{organ}', tipo='{ans_type_str}'")

            with st.spinner("Analisi in corso..."):
                pixel_values, input_ids, _ = preprocess(
                    image, question, organ, q_type, tokenizer, image_processor
                )
                q_type_tensor = torch.tensor([q_type_idx], dtype=torch.long).to(DEVICE)

                generated_ids, confidences = model.generate(
                    pixel_values=pixel_values,
                    question_ids=input_ids,
                    question_types=q_type_tensor,
                    tokenizer=tokenizer,
                    num_beams=5,
                    max_len=32,
                )

                answer     = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
                confidence = confidences[0].item()

            # ── Gestione risposta vuota ────────────────────────────────────
            if not answer:
                answer = "N/A"
                st.markdown("""
                <div style="background:#21262d;border:1.5px solid #d29922;border-radius:8px;
                            padding:0.8rem 1.2rem;margin-bottom:0.5rem;font-size:0.88rem;color:#d29922;">
                    ⚠️ Il modello non ha generato una risposta interpretabile. 
                    Prova a riformulare la domanda.
                </div>
                """, unsafe_allow_html=True)

            # ── Risposta ──────────────────────────────────────────────────
            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-label">Risposta del modello</div>
                <div class="answer-text">{answer}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Metriche ──────────────────────────────────────────────────
            conf_pct    = confidence * 100
            threshold   = (CONFIDENCE_THRESHOLD_CLOSED if ans_type_str == "CLOSED"
                           else CONFIDENCE_THRESHOLD_OPEN) * 100
            conf_class  = "good" if conf_pct >= threshold else ("warn" if conf_pct >= threshold * 0.7 else "bad")
            bar_color   = {"good": "#3fb950", "warn": "#d29922", "bad": "#f85149"}[conf_class]

            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value {conf_class}">{conf_pct:.1f}%</div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill"
                             style="width:{conf_pct:.1f}%; background:{bar_color};">
                        </div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tipo risposta</div>
                    <div class="metric-value">{ans_type_str}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Organo rilevato</div>
                    <div class="metric-value">{organ}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Categoria domanda</div>
                    <div class="metric-value">{q_type}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if conf_pct < threshold:
                st.markdown(f"""
                <div style="background:#21262d;border:1.5px solid #d29922;border-radius:8px;
                            padding:0.8rem 1.2rem;margin-top:0.5rem;font-size:0.88rem;color:#d29922;">
                    ⚠️ Confidence sotto soglia ({threshold:.0f}%). 
                    Verificare con un radiologo.
                </div>
                """, unsafe_allow_html=True)

            # ── Spiegazione Gemini (skip se answer è N/A) ─────────────────
            if answer != "N/A":
                st.markdown("<hr>", unsafe_allow_html=True)
                with st.spinner("Gemini sta elaborando la spiegazione clinica..."):
                    explanation = gemini_explain(
                        question=question,
                        answer=answer,
                        organ=organ,
                        q_type=q_type,
                        ans_type=ans_type_str,
                        confidence=confidence,
                    )

                st.markdown(f"""
                <div class="gemini-card">
                    <div class="gemini-header">✦ Spiegazione clinica · Gemini {GEMINI_MODEL}</div>
                    <div class="gemini-text">{explanation}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        # Placeholder quando non c'è ancora un risultato
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;justify-content:center;
                    align-items:center;padding:4rem 2rem;opacity:0.35;text-align:center;">
            <div style="font-size:4rem;margin-bottom:1rem;">🩻</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.75rem;
                        letter-spacing:2px;text-transform:uppercase;">
                Carica un'immagine e scrivi<br>una domanda per iniziare
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#30363d;
            text-align:center;letter-spacing:1px;">
    MEDVQA · Bio_ClinicalBERT + ViT-Large · VQA-RAD Dataset · 
    Solo per uso di ricerca — non per uso clinico diagnostico
</div>
""", unsafe_allow_html=True)