"""
AnalisiEsplorativa.py — Analisi Esplorativa (EDA) del dataset VQA-RAD
=====================================================================
Questo script esegue un'analisi esplorativa completa del dataset di Medical
Visual Question Answering VQA-RAD, salvato in formato HuggingFace/Arrow nella
cartella "dataset/VQA_RAD Output Folder" (split: train e test).

Obiettivo: capire com'è fatto il dataset PRIMA di addestrare il modello,
mettendo in evidenza eventuali sbilanciamenti (es. CLOSED vs OPEN, organi
sovra/sotto-rappresentati, risposte ricorrenti come 'yes'/'no') che possono
influenzare le scelte di training e l'interpretazione delle metriche.

Cosa fa:
  1. Carica i file .arrow dei due split SENZA caricare i byte delle immagini
     (la colonna 'image' è pesante e qui non serve), usando solo pyarrow:
     niente dipendenza dalla libreria 'datasets'.
  2. Pulisce i campi testuali (strip + uppercase su etichette categoriche) per
     evitare doppioni fantasma come 'CLOSED' vs 'CLOSED ' (con spazio finale).
  3. Calcola statistiche descrittive e le stampa a console.
  4. Genera una serie di grafici e li salva in PNG ad alta risoluzione nella
     cartella ./analisi_esplorativa_plots.

Grafici prodotti:
  - Numero di campioni per split (train/test)
  - Bilanciamento answer_type (CLOSED vs OPEN), globale e per split
  - Distribuzione per organo (image_organ)
  - Distribuzione question_type (tipi di domanda)
  - Distribuzione phrase_type (frasi originali vs parafrasate)
  - Top-N risposte più frequenti
  - Bilanciamento risposte sì/no (le CLOSED binarie)
  - Distribuzione della lunghezza delle domande (in parole)
  - Distribuzione del numero di domande per immagine
  - Heatmap organo x answer_type

Uso:
    python AnalisiEsplorativa.py
"""

import os
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURAZIONE PERCORSI
# ==========================================
# Cartella del dataset Arrow (relativa a codice/, con fallback assoluto).
DATASET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "dataset", "VQA_RAD Output Folder")
)
if not os.path.isdir(DATASET_DIR):
    DATASET_DIR = r"C:/Users/angel/OneDrive/Desktop/ProgettoNLP/progetto/dataset/VQA_RAD Output Folder"

# Cartella di output dei grafici: creata se non esiste.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analisi_esplorativa_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stile grafico coerente per tutti i plot.
sns.set_theme(style="whitegrid")
PALETTE = "viridis"


# ==========================================
# CARICAMENTO DATI
# ==========================================
def carica_split(split):
    """
    Carica un singolo split (train/test) del dataset Arrow in un DataFrame pandas,
    escludendo la colonna 'image' (byte delle immagini) per non sprecare memoria.

    Legge il file .arrow tramite pyarrow.ipc su memoria mappata, seleziona tutte
    le colonne tranne 'image' e converte in pandas. Aggiunge una colonna 'split'
    per poter poi concatenare train e test mantenendo la provenienza.

    Args:
        split: nome dello split, "train" o "test"

    Returns:
        pandas.DataFrame con le colonne testuali/categoriche + 'split'
    """
    path = os.path.join(DATASET_DIR, split, "data-00000-of-00001.arrow")
    with pa.memory_map(path) as src:
        tabella = ipc.open_stream(src).read_all()

    colonne = [c for c in tabella.column_names if c != "image"]
    df = tabella.select(colonne).to_pandas()
    df["split"] = split
    return df


def pulisci(df):
    """
    Normalizza i campi categorici testuali per eliminare doppioni dovuti a spazi
    o differenze di maiuscole/minuscole (es. 'CLOSED ' -> 'CLOSED').

    Applica strip() a tutte le stringhe e uppercase alle etichette categoriche
    (organo, tipo risposta, tipo domanda, tipo frase). Aggiunge inoltre due
    colonne derivate utili per l'analisi:
      - 'q_len_parole': numero di parole della domanda
      - 'answer_norm':  risposta in minuscolo e senza spazi ai bordi

    Args:
        df: DataFrame grezzo da carica_split

    Returns:
        DataFrame pulito con le colonne derivate
    """
    df = df.copy()
    for col in ["image_organ", "answer_type", "question_type", "phrase_type"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    for col in ["question", "answer", "image_name"]:
        df[col] = df[col].astype(str).str.strip()

    df["q_len_parole"] = df["question"].str.split().apply(len)
    df["answer_norm"] = df["answer"].str.lower().str.strip()
    return df


# ==========================================
# STATISTICHE A CONSOLE
# ==========================================
def stampa_statistiche(df):
    """
    Stampa a console un riepilogo testuale delle statistiche principali del
    dataset: dimensioni, valori unici, bilanciamenti e indicatori di squilibrio.

    Serve come referto rapido da copiare nella relazione senza dover aprire i
    grafici. Evidenzia in particolare il rapporto CLOSED/OPEN e la quota di
    risposte sì/no, i due squilibri tipici di VQA-RAD.

    Args:
        df: DataFrame completo (train+test) già pulito
    """
    print("=" * 70)
    print("ANALISI ESPLORATIVA — DATASET VQA-RAD")
    print("=" * 70)
    print(f"Campioni totali        : {len(df)}")
    print(f"  - train              : {(df['split'] == 'train').sum()}")
    print(f"  - test               : {(df['split'] == 'test').sum()}")
    print(f"Immagini uniche        : {df['image_name'].nunique()}")
    print(f"Domande uniche         : {df['question'].nunique()}")
    print(f"Risposte uniche        : {df['answer_norm'].nunique()}")
    print(f"Organi                 : {sorted(df['image_organ'].unique())}")
    print(f"Tipi di risposta       : {sorted(df['answer_type'].unique())}")
    print(f"Tipi di domanda        : {df['question_type'].nunique()} categorie")

    print("\n--- Bilanciamento answer_type (globale) ---")
    vc = df["answer_type"].value_counts()
    for k, v in vc.items():
        print(f"  {k:8s}: {v:5d}  ({v / len(df) * 100:5.1f}%)")
    if {"CLOSED", "OPEN"}.issubset(vc.index):
        print(f"  Rapporto CLOSED/OPEN: {vc['CLOSED'] / vc['OPEN']:.2f}")

    print("\n--- Risposte più frequenti (top 10) ---")
    for k, v in df["answer_norm"].value_counts().head(10).items():
        print(f"  {k:25s}: {v:4d}  ({v / len(df) * 100:4.1f}%)")

    yes_no = df["answer_norm"].isin(["yes", "no"]).sum()
    print(f"\nRisposte sì/no         : {yes_no}  ({yes_no / len(df) * 100:.1f}% del totale)")
    print(f"Lunghezza domanda (parole): media={df['q_len_parole'].mean():.1f}, "
          f"min={df['q_len_parole'].min()}, max={df['q_len_parole'].max()}")
    print("=" * 70)


# ==========================================
# FUNZIONI DI PLOT
# ==========================================
def _salva(nome):
    """Salva la figura corrente in OUTPUT_DIR a 300 DPI e la chiude."""
    path = os.path.join(OUTPUT_DIR, nome)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[salvato] {path}")


def _annota_barre(ax, percentuale_su=None):
    """
    Scrive il valore numerico sopra ogni barra di un barplot.

    Se percentuale_su è valorizzato (totale dei campioni), aggiunge anche la
    percentuale tra parentesi, comoda per leggere a colpo d'occhio quanto pesa
    ogni categoria sul totale.
    """
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            testo = f"{int(h)}"
            if percentuale_su:
                testo += f"\n({h / percentuale_su * 100:.1f}%)"
            ax.annotate(testo, (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9)


def plot_campioni_per_split(df):
    """Bar chart del numero di campioni in train vs test."""
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="split", hue="split",
                       palette=PALETTE, legend=False,
                       order=["train", "test"])
    ax.set_title("Numero di campioni per split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Numero di campioni")
    _annota_barre(ax, percentuale_su=len(df))
    _salva("01_campioni_per_split.png")


def plot_answer_type(df):
    """
    Bilanciamento CLOSED vs OPEN: un grafico globale e uno separato per split.
    È il grafico chiave per capire se/quanto il dataset è sbilanciato.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sns.countplot(data=df, x="answer_type", hue="answer_type",
                  palette=PALETTE, legend=False,
                  order=df["answer_type"].value_counts().index, ax=ax)
    ax.set_title("Bilanciamento answer_type (globale)")
    ax.set_xlabel("Tipo di risposta")
    ax.set_ylabel("Numero di campioni")
    _annota_barre(ax, percentuale_su=len(df))

    ax = axes[1]
    sns.countplot(data=df, x="answer_type", hue="split",
                  palette="Set2", ax=ax,
                  order=df["answer_type"].value_counts().index)
    ax.set_title("answer_type per split")
    ax.set_xlabel("Tipo di risposta")
    ax.set_ylabel("Numero di campioni")

    fig.suptitle("OPEN vs CLOSED — verifica dello sbilanciamento", fontsize=14)
    _salva("02_answer_type_bilanciamento.png")


def plot_organo(df):
    """Distribuzione dei campioni per organo anatomico (CHEST, HEAD, ABD)."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="image_organ", hue="image_organ",
                       palette=PALETTE, legend=False,
                       order=df["image_organ"].value_counts().index)
    ax.set_title("Distribuzione per organo (image_organ)")
    ax.set_xlabel("Organo")
    ax.set_ylabel("Numero di campioni")
    _annota_barre(ax, percentuale_su=len(df))
    _salva("03_distribuzione_organi.png")


def plot_question_type(df):
    """Distribuzione dei tipi di domanda (question_type), ordinata e orizzontale."""
    plt.figure(figsize=(9, 8))
    ordine = df["question_type"].value_counts().index
    ax = sns.countplot(data=df, y="question_type", hue="question_type",
                       palette=PALETTE, legend=False, order=ordine)
    ax.set_title("Distribuzione dei tipi di domanda (question_type)")
    ax.set_xlabel("Numero di campioni")
    ax.set_ylabel("Tipo di domanda")
    _salva("04_question_type.png")


def plot_phrase_type(df):
    """Distribuzione phrase_type (frasi originali vs parafrasate/freeform)."""
    plt.figure(figsize=(8, 5))
    ordine = df["phrase_type"].value_counts().index
    ax = sns.countplot(data=df, x="phrase_type", hue="phrase_type",
                       palette=PALETTE, legend=False, order=ordine)
    ax.set_title("Distribuzione phrase_type")
    ax.set_xlabel("Tipo di frase")
    ax.set_ylabel("Numero di campioni")
    ax.tick_params(axis="x", rotation=20)
    _annota_barre(ax, percentuale_su=len(df))
    _salva("05_phrase_type.png")


def plot_top_risposte(df, n=20):
    """
    Top-N risposte più frequenti. Mette in luce quanto il dataset sia dominato
    da poche risposte (tipicamente 'yes'/'no'), un altro segnale di squilibrio.
    """
    plt.figure(figsize=(9, 9))
    top = df["answer_norm"].value_counts().head(n)
    ax = sns.barplot(x=top.values, y=top.index, hue=top.index,
                     palette=PALETTE, legend=False)
    ax.set_title(f"Top {n} risposte più frequenti")
    ax.set_xlabel("Frequenza")
    ax.set_ylabel("Risposta")
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    _salva("06_top_risposte.png")


def plot_yes_no(df):
    """
    Bilanciamento sì/no all'interno delle risposte binarie, confrontato con
    tutte le altre risposte. Utile per quantificare il bias verso il 'yes'.
    """
    cat = df["answer_norm"].apply(
        lambda a: a if a in ("yes", "no") else "altro")
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=cat, hue=cat, palette=PALETTE, legend=False,
                       order=cat.value_counts().index)
    ax.set_title("Risposte sì / no / altro")
    ax.set_xlabel("Risposta")
    ax.set_ylabel("Numero di campioni")
    _annota_barre(ax, percentuale_su=len(df))
    _salva("07_yes_no.png")


def plot_lunghezza_domande(df):
    """Istogramma della lunghezza delle domande (in numero di parole)."""
    plt.figure(figsize=(9, 5))
    ax = sns.histplot(data=df, x="q_len_parole", hue="answer_type",
                      bins=range(0, df["q_len_parole"].max() + 2),
                      multiple="stack", palette="Set2")
    ax.set_title("Distribuzione della lunghezza delle domande (parole)")
    ax.set_xlabel("Numero di parole nella domanda")
    ax.set_ylabel("Numero di campioni")
    ax.axvline(df["q_len_parole"].mean(), color="red", linestyle="--",
               label=f"media = {df['q_len_parole'].mean():.1f}")
    ax.legend()
    _salva("08_lunghezza_domande.png")


def plot_domande_per_immagine(df):
    """
    Istogramma del numero di domande associate a ciascuna immagine: indica
    quanto è "denso" il dataset per immagine e se poche immagini concentrano
    molte domande.
    """
    per_img = df.groupby("image_name").size()
    plt.figure(figsize=(9, 5))
    ax = sns.histplot(per_img, bins=range(0, per_img.max() + 2), color="teal")
    ax.set_title("Numero di domande per immagine")
    ax.set_xlabel("Domande associate alla stessa immagine")
    ax.set_ylabel("Numero di immagini")
    ax.axvline(per_img.mean(), color="red", linestyle="--",
               label=f"media = {per_img.mean():.1f}")
    ax.legend()
    _salva("09_domande_per_immagine.png")


def plot_heatmap_organo_answertype(df):
    """
    Heatmap organo x answer_type: mostra come si distribuiscono OPEN/CLOSED
    dentro ciascun organo, rivelando squilibri incrociati (es. un organo con
    quasi solo domande CLOSED).
    """
    tab = pd.crosstab(df["image_organ"], df["answer_type"])
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(tab, annot=True, fmt="d", cmap=PALETTE)
    ax.set_title("Campioni per organo x tipo di risposta")
    ax.set_xlabel("Tipo di risposta")
    ax.set_ylabel("Organo")
    _salva("10_heatmap_organo_answertype.png")


# ==========================================
# ENTRY POINT
# ==========================================
def main():
    """
    Pipeline completa: carica i due split, li pulisce e concatena, stampa le
    statistiche e genera tutti i grafici nella cartella di output.
    """
    print(f"Lettura dataset da: {DATASET_DIR}")
    train = pulisci(carica_split("train"))
    test = pulisci(carica_split("test"))
    df = pd.concat([train, test], ignore_index=True)

    stampa_statistiche(df)

    plot_campioni_per_split(df)
    plot_answer_type(df)
    plot_organo(df)
    plot_question_type(df)
    plot_phrase_type(df)
    plot_top_risposte(df)
    plot_yes_no(df)
    plot_lunghezza_domande(df)
    plot_domande_per_immagine(df)
    plot_heatmap_organo_answertype(df)

    print(f"\nFatto. {len(os.listdir(OUTPUT_DIR))} grafici in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
