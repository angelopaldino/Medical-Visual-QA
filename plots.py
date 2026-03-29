"""
plots.py — Visualizzazione dei risultati degli esperimenti di training Medical VQA
===================================================================================
Questo script raccoglie tutte le funzioni di plotting usate per analizzare e confrontare
i risultati dei vari addestramenti del modello VQA. L'obiettivo è produrre grafici
chiari e leggibili che mettano in evidenza i progressi tra un esperimento e l'altro,
sia a livello globale che disaggregato per organo e tipo di risposta.

Funzioni principali:
  1. parse_files(file_list):
       Legge i file di log .txt degli addestramenti ed estrae con regex le tre
       metriche finali (Exact Match, BLEU-1, BERTScore F1). Usata come step
       preliminare prima di plot_inference_metrics.

  2. plot_inference_metrics(data):
       Bar chart a gruppi che confronta le tre metriche su tutti gli esperimenti
       con risultati di inferenza disponibili. Ogni esperimento ha tre barre
       affiancate (una per metrica), con annotazioni numeriche sopra ciascuna.

  3. mostra_tabella_grafica(domande, predizioni, ground_truths):
       Genera un'immagine con una tabella visiva che mostra, per un campione
       di esempi, la domanda, la predizione del modello, la ground truth e
       l'esito (Corretta / Errata). Usa textwrap.fill per evitare che le
       parole vengano spezzate a capo.

  4. confronta_tutte_metriche_organi(lista_organi, dati_esperimenti):
       Tre subplot affiancati (uno per metrica) che confrontano le performance
       per organo (CHEST, HEAD, ABD) tra più esperimenti. Stampa anche una
       tabella riassuntiva in formato markdown.

  5. confronta_metriche_tipo_risposta(tipi_risposta, dati_esperimenti):
       Stessa struttura della funzione precedente, ma disaggregata per tipo
       di risposta (OPEN vs CLOSED) anziché per organo.

Il blocco __main__ contiene i dati hardcoded degli esperimenti finali e le
chiamate alle funzioni (alcune commentate per non rieseguirle ogni volta).
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

# Lista dei file di log degli addestramenti, in ordine cronologico.
# Ogni file .txt contiene i log di un singolo run di training, con le metriche
# finali stampate dal notebook medicalvqa.ipynb al termine della valutazione.
files = [
    "./addestramento.txt", "./addestramento2.txt", "./addestramento3.txt",
    "./addestramento4.txt", "./addestramento5.txt", "./Addestramento6.txt",
    "./Addestramento7.txt", "./Addestramento8.txt", "./Addestramento9.txt",
    "./AddestramentoFinale.txt"
]

def parse_files(file_list):
    """
    Legge i file di log degli addestramenti ed estrae le metriche finali di inferenza.

    Per ogni file presente su disco, cerca con espressioni regolari tre valori:
      - Exact Match Accuracy: percentuale di risposte identiche alla ground truth
      - BLEU-1: punteggio di sovrapposizione lessicale (unigrammi)
      - BERTScore F1: punteggio di similarità semantica

    Le regex gestiscono varianti di nomenclatura nei diversi file di log
    (es. 'Exact Match Accuracy' vs 'Exact Match Accuracy Globale').
    Se un valore non viene trovato, viene impostato a 0.0 per sicurezza.

    Args:
        file_list: lista di percorsi ai file .txt dei log di training

    Returns:
        dict: chiave = percorso file, valore = dict con 'exact_match', 'bleu1', 'bert_f1'
    """
    data = {}
    for f in file_list:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()                
                # Trova le metriche di test finali gestendo varie nomenclature nei file
                exact_match = re.search(r'Exact Match Accuracy(?: Globale)?:\s*([0-9.]+)', content)
                bleu = re.search(r'(?:Average )?BLEU-1(?: Score| Globale)?:\s*([0-9.]+)', content)
                bert_f1 = re.search(r'BERTScore F1(?: Globale)?:\s*([0-9.]+)', content)
                
                data[f] = {
                    'exact_match': float(exact_match.group(1)) if exact_match else 0.0,
                    'bleu1': float(bleu.group(1)) if bleu else 0.0,
                    'bert_f1': float(bert_f1.group(1)) if bert_f1 else 0.0
                }
    return data


def plot_inference_metrics(data):
    """
    Genera un bar chart a gruppi che confronta Exact Match, BLEU-1 e BERTScore F1
    tra tutti gli esperimenti che hanno prodotto risultati di inferenza.

    Per ogni esperimento vengono disegnate tre barre affiancate (una per metrica),
    con annotazioni numeriche sopra ogni barra per leggibilità immediata.
    Gli esperimenti senza metriche (es. addestramenti interrotti prima della valutazione)
    vengono automaticamente esclusi dal grafico.

    Il grafico viene salvato su disco come 'inference_metrics_comparison.png' a 300 DPI.

    Args:
        data: output di parse_files — dict {filename: {exact_match, bleu1, bert_f1}}
    """
    labels = []
    exact_matches = []
    bleus = []
    bert_scores = []
    
    for filename, metrics in data.items():
        # Scartiamo i file che non hanno risultati di inferenza, come l'addestramento 2
        if metrics['exact_match'] > 0 or metrics['bleu1'] > 0 or metrics['bert_f1'] > 0:
            labels.append(filename.replace('.txt', '').replace('Esperimento', 'Add. '))
            exact_matches.append(metrics['exact_match'])
            bleus.append(metrics['bleu1'])
            bert_scores.append(metrics['bert_f1'])
            
    x = np.arange(len(labels))
    width = 0.25 # Larghezza delle barre
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width, exact_matches, width, label='Exact Match Accuracy', color='skyblue')
    rects2 = ax.bar(x, bleus, width, label='BLEU-1', color='lightgreen')
    rects3 = ax.bar(x + width, bert_scores, width, label='BERTScore F1', color='salmon')
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Confronto delle Metriche di Inferenza (BERTScore, Exact Match, BLEU)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Aggiungiamo le annotazioni sopra ogni barra
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
                            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    # Set ylim un pelo più alto del max score per evitare sovrapposizioni delle scritte
    ax.set_ylim(0, 1.05)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    plt.savefig('inference_metrics_comparison.png', dpi=300)
    plt.show()



def mostra_tabella_grafica(domande, predizioni, ground_truths):
    """
    Genera e salva un'immagine con una tabella visiva delle predizioni del modello.

    Per ogni esempio mostra: domanda, risposta del modello, ground truth e
    un esito colorato (verde = Corretta, rosso = Errata).

    Usa textwrap.fill() per gestire il word wrapping senza spezzare le parole a metà,
    assegnando larghezze in caratteri diverse per ogni colonna (domanda più larga,
    esito più stretto). Le celle vengono scalate in altezza per contenere il testo
    multiriga senza sovrapposizioni.

    Il risultato viene salvato come 'tabella_predizioni_corretta.png' a 300 DPI
    con bbox_inches='tight' per non tagliare i bordi della tabella.

    Args:
        domande: lista di stringhe con le domande cliniche
        predizioni: lista di stringhe con le risposte generate dal modello
        ground_truths: lista di stringhe con le risposte corrette
    """
    # 1. Calcolo degli esiti
    esiti = []
    for p, gt in zip(predizioni, ground_truths):
        if p.strip().lower() == gt.strip().lower():
            esiti.append("Corretta")
        else:
            esiti.append("Errata")
            
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Analisi Esempi di Predizione vs Ground Truth', fontsize=18, fontweight='bold', y=0.95)
    
    # Rimuoviamo gli assi
    ax.axis('tight')
    ax.axis('off') 
    
    # 2. Formattazione intelligente del testo per evitare parole spezzate
    table_data = []
    for d, p, gt, esito in zip(domande, predizioni, ground_truths, esiti):
        # textwrap.fill va a capo automaticamente mantenendo le parole intatte!
        # Impostiamo le larghezze in caratteri per colonna
        d_wrap = textwrap.fill(d, width=35)
        p_wrap = textwrap.fill(p, width=35)
        gt_wrap = textwrap.fill(gt, width=35)
        
        table_data.append([d_wrap, p_wrap, gt_wrap, esito])
        
    col_labels = ['Domanda', 'Predizione Modello', 'Ground Truth (GT)', 'Esito']
    
    # 3. Creazione tabella con larghezze colonne personalizzate (percentuali)
    # diamo più spazio alla Domanda (35%) e alla Ground Truth (30%)
    tabella = ax.table(
        cellText=table_data, 
        colLabels=col_labels, 
        loc='center', 
        cellLoc='center', 
        colWidths=[0.35, 0.25, 0.30, 0.10] 
    )
    
    tabella.auto_set_font_size(False)
    tabella.set_fontsize(11)
    
    # Allarghiamo l'altezza delle celle per contenere tutto il testo multiriga comodamente
    tabella.scale(1, 4.5) 
    
    # 4. Colorazione delle celle
    for (row, col), cell in tabella.get_celld().items():
        if row == 0:
            # Stile dell'intestazione (verde scuro)
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white', fontweight='bold', fontsize=12)
        else:
            # Colorazione dell'esito (verde/rosso chiaro)
            if col == 3:
                if "Corretta" in cell.get_text().get_text():
                    cell.set_facecolor('#e6ffe6') 
                else:
                    cell.set_facecolor('#ffe6e6') 

    plt.tight_layout()
    
    # Salvataggio dell'immagine con bbox_inches='tight' per non tagliare i bordi
    plt.savefig('tabella_predizioni_corretta.png', dpi=300, bbox_inches='tight')
    plt.show()


    
    


def confronta_tutte_metriche_organi(lista_organi, dati_esperimenti):
    """
    Confronta le performance del modello per organo anatomico tra più esperimenti.

    Genera tre subplot affiancati (uno per ciascuna metrica: Exact Match, BLEU-1,
    BERTScore F1). All'interno di ogni subplot, le barre sono raggruppate per organo,
    e ogni gruppo mostra le barre dei diversi esperimenti side-by-side con colori
    distinti dalla colormap 'tab10'.

    Stampa anche una tabella riassuntiva in formato markdown con tutti i valori,
    utile per una verifica numerica rapida.

    La larghezza delle barre è calcolata dinamicamente in base al numero di esperimenti
    per evitare sovrapposizioni indipendentemente da quanti run vengono confrontati.
    La legenda appare solo nel primo subplot per non appesantire la visualizzazione.

    Args:
        lista_organi: lista di nomi degli organi (es. ["CHEST", "HEAD", "ABD"])
        dati_esperimenti: dict annidato {nome_esperimento: {metrica: [val_per_organo]}}
    """
    metriche_da_plot = ["Exact Match", "BLEU-1", "BERTScore F1"]
    
    # 1. Creazione e Stampa della Tabella riassuntiva completa
    righe_tabella = []
    for esperimento, metriche in dati_esperimenti.items():
        for i, organo in enumerate(lista_organi):
            riga = {
                "Esperimento": esperimento,
                "Organo": organo,
                "Exact Match": metriche["Exact Match"][i],
                "BLEU-1": metriche["BLEU-1"][i],
                "BERTScore F1": metriche["BERTScore F1"][i]
            }
            righe_tabella.append(riga)
            
    df = pd.DataFrame(righe_tabella)
    print("\n--- Tabella Completa Multi-Esperimento e Multi-Metrica ---")
    print(df.to_markdown(index=False))
    
    # 2. Generazione dei 3 Grafici Affiancati (Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confronto Performance per Organo (Multi-Esperimento)', fontsize=18, y=1.05)
    
    x = np.arange(len(lista_organi))
    num_esperimenti = len(dati_esperimenti)
    width = 0.8 / num_esperimenti  # Larghezza dinamica delle barre
    
    # Colori personalizzati per gli esperimenti (per coerenza tra i grafici)
    colori = plt.cm.get_cmap('tab10', num_esperimenti)
    
    for idx_metrica, metrica in enumerate(metriche_da_plot):
        ax = axes[idx_metrica]
        
        for i, (nome_esperimento, metriche_exp) in enumerate(dati_esperimenti.items()):
            valori = metriche_exp[metrica]
            offset = (i - num_esperimenti / 2) * width + width / 2
            
            rects = ax.bar(x + offset, valori, width, label=nome_esperimento, color=colori(i))
            
            # Aggiunge i valori testuali sopra le barre
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), 
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8)

        # Setup del singolo grafico
        ax.set_title(f'Metrica: {metrica}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(lista_organi, fontsize=11)
        ax.set_ylabel('Punteggio')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Aumenta il limite Y in base al massimo valore di quella metrica
        max_val = max([max(exp[metrica]) for exp in dati_esperimenti.values() if exp[metrica]])
        ax.set_ylim(0, max_val + 0.15)
        
        # Mette la legenda solo nel primo grafico per non ripeterla inutilmente
        if idx_metrica == 0:
            ax.legend(title="Esperimenti", loc='upper left')

    plt.tight_layout()
    plt.show()


def visualize_query_3d(collection, encoder, query_text, title="Visualizzazione 3D"):
# 1. Recuperiamo i dati esistenti
    data = collection.get(include=['embeddings', 'metadatas', 'documents'])
    existing_embeddings = np.array(data['embeddings'])
        
    # 2. Trasformiamo la NUOVA domanda in un vettore (Embedding)
    query_embedding = encoder.encode([query_text], normalize_embeddings=True)
        
    # 3. IL RESTRINGITORE (PCA)
    # Importante: alleniamo il PCA su TUTTI i dati per creare lo spazio
    pca = PCA(n_components=3)
    pca.fit(existing_embeddings) 
        
    # Trasformiamo sia i dati vecchi che la nuova domanda nelle stesse 3 coordinate
    coords_existing = pca.transform(existing_embeddings)
    coords_query = pca.transform(query_embedding)

    # 4. Prepariamo la tabella per i dati esistenti
    df_existing = pd.DataFrame({
        'x': coords_existing[:, 0], 'y': coords_existing[:, 1], 'z': coords_existing[:, 2],
        'Etichetta': [m.get('organ', m.get('source', 'N/A')) for m in data['metadatas']],
        'Testo': [d[:100] for d in data['documents']],
        'Tipo': 'Archivio'
    })

    # 5. Prepariamo la tabella per il Punto Rosso
    df_query = pd.DataFrame({
        'x': [coords_query[0, 0]], 'y': [coords_query[0, 1]], 'z': [coords_query[0, 2]],
        'Etichetta': ['DOMANDA NUOVA'],
        'Testo': [query_text],
        'Tipo': 'Query'
    })

    # Uniamo tutto
    df_final = pd.concat([df_existing, df_query])

    # 6. Disegniamo il grafico
    fig = px.scatter_3d(
        df_final, x='x', y='y', z='z',
        color='Etichetta',
        symbol='Tipo', # Cambia forma tra archivio e query
        hover_data=['Testo'],
        title=title,
        template="plotly_dark"
    )

    fig.update_traces(marker=dict(size=10), selector=dict(name='DOMANDA NUOVA'))
    fig.show()
        


def confronta_metriche_tipo_risposta(tipi_risposta, dati_esperimenti):
    """
    Confronta le performance del modello per tipo di risposta (OPEN vs CLOSED)
    tra più esperimenti.

    Stessa struttura di confronta_tutte_metriche_organi, ma l'asse X mostra
    i tipi di risposta anziché gli organi. Usa la colormap 'Set2' per variare
    visivamente rispetto all'altra funzione.

    Questo confronto è particolarmente rivelatore: il modello tende ad avere
    performance molto più alte sulle domande CLOSED (sì/no) rispetto a quelle
    OPEN (risposte descrittive), ed è utile monitorare se questa disparità
    si riduce tra un esperimento e l'altro.

    Args:
        tipi_risposta: lista dei tipi di risposta (es. ["OPEN", "CLOSED"])
        dati_esperimenti: dict annidato {nome_esperimento: {metrica: [val_per_tipo]}}
    """
    metriche_da_plot = ["Exact Match", "BLEU-1", "BERTScore F1"]
    
    # 1. Creazione e Stampa della Tabella riassuntiva
    righe_tabella = []
    for esperimento, metriche in dati_esperimenti.items():
        for i, tipo in enumerate(tipi_risposta):
            riga = {
                "Esperimento": esperimento,
                "Tipo Risposta": tipo,
                "Exact Match": metriche["Exact Match"][i],
                "BLEU-1": metriche["BLEU-1"][i],
                "BERTScore F1": metriche["BERTScore F1"][i]
            }
            righe_tabella.append(riga)
            
    df = pd.DataFrame(righe_tabella)
    print("\n--- Tabella Performance per Tipo di Risposta (OPEN vs CLOSED) ---")
    print(df.to_markdown(index=False))
    
    # 2. Generazione dei 3 Grafici Affiancati
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Confronto Performance per Tipo di Risposta (OPEN vs CLOSED)', fontsize=18, y=1.05)
    
    x = np.arange(len(tipi_risposta))
    num_esperimenti = len(dati_esperimenti)
    width = 0.8 / num_esperimenti  
    
    colori = plt.cm.get_cmap('Set2', num_esperimenti) # Usiamo una palette diversa per variare
    
    for idx_metrica, metrica in enumerate(metriche_da_plot):
        ax = axes[idx_metrica]
        
        for i, (nome_esperimento, metriche_exp) in enumerate(dati_esperimenti.items()):
            valori = metriche_exp[metrica]
            offset = (i - num_esperimenti / 2) * width + width / 2
            
            rects = ax.bar(x + offset, valori, width, label=nome_esperimento, color=colori(i))
            
            # Aggiunge i valori sopra le barre
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), 
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

        # Setup del singolo grafico
        ax.set_title(f'{metrica}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(tipi_risposta, fontsize=12, fontweight='bold')
        ax.set_ylabel('Punteggio')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Aumenta il limite Y
        max_val = max([max(exp[metrica]) for exp in dati_esperimenti.values() if exp[metrica]])
        ax.set_ylim(0, max_val + 0.15)
        
        # Legenda solo nel primo grafico
        if idx_metrica == 0:
            ax.legend(title="Esperimenti", loc='upper left')

    plt.tight_layout()
    plt.show()



    


# ==========================================
# ENTRY POINT
#
# Il blocco principale esegue le funzioni di plotting con i dati hardcoded
# degli esperimenti finali. Alcune chiamate sono commentate per non
# rieseguire tutti i plot ogni volta — decommentare quelle di interesse.
#
# Struttura dei dati hardcoded:
#   - dati_completi: metriche per organo (CHEST, HEAD, ABD) degli addestramenti 6-10
#   - dati_tipo_risposta: metriche per tipo (OPEN, CLOSED) degli addestramenti 6-9
#   - domande/preds/gts: campione di 6 esempi per la tabella predizioni
# ==========================================
if __name__ == "__main__":

    #dati_estratti = parse_files(files)
    #plot_inference_metrics(dati_estratti)
    
    # Dati per la prima funzione
    domande = ["is there evidence of an aortic aneurysm?","is there airspace consolidation on the left side?","is there any intraparenchymal abnormalities in the lung fields?","What side is the pathology?","Describe the size of this lesion?","How do you determine cardiomegaly?"]
    preds = ["yes", "yes", "yes","right sided pleural effusion","5. 6cm focal, predominantly hypodense","right"]
    gts = ["yes", "yes","no","right sided pleural effusion","5. 6cm focal, predominantly hypodense","if the heart diameter is greater than half the diameter of the thoracic cavity"]
    #mostra_tabella_grafica(domande,preds, gts)

    
    # Dati per la seconda funzione
    lista_organi = ["CHEST", "HEAD", "ABD"]
    dati_completi = {
        "Addestramento 6": {
            "Exact Match": [0.4762, 0.3846, 0.4444],
            "BLEU-1":      [0.4942, 0.4023, 0.4466],
            "BERTScore F1":[0.8674, 0.8283, 0.8429]
        },
        "Addestramento 7": {
            "Exact Match": [0.5000, 0.3173, 0.4630],
            "BLEU-1":      [0.5127, 0.3205, 0.4630],
            "BERTScore F1":[0.8506, 0.7693, 0.8128]
        },
        "Addestramento 8": {
            "Exact Match": [0.5476, 0.3750, 0.4259],
            "BLEU-1":      [0.5544, 0.3946, 0.4290],
            "BERTScore F1":[0.8774, 0.8302, 0.8360]
        },
        "Addestramento 9": {
            "Exact Match": [0.2857, 0.1731, 0.2222],
            "BLEU-1":      [0.2892, 0.1779, 0.2222],
            "BERTScore F1":[0.8216, 0.7590, 0.8070]
        },
        "Addestramento 10": {
            "Exact Match": [0.4921, 0.3462, 0.4630],
            "BLEU-1":      [0.5151, 0.3673, 0.4760],
            "BERTScore F1":[0.8637, 0.8178, 0.8528]
        }
    }
    #confronta_tutte_metriche_organi(lista_organi, dati_completi)

    tipi = ["OPEN", "CLOSED"]
    
    # Dizionario Annidato: l'ordine delle liste corrisponde a ["OPEN", "CLOSED"]
    # Valori di esempio basati sui log tipici per le VQA (le closed hanno sempre accuracy molto più alta)
    dati_tipo_risposta = {
        "Addestramento 6": {
            "Exact Match": [0.1295, 0.6533], 
            "BLEU-1":      [0.1607, 0.6533], 
            "BERTScore F1":[0.7185, 0.9377]
        },
        "Addestramento 7": {
            "Exact Match": [0.0072, 0.7286], 
            "BLEU-1":      [0.0211, 0.7286], 
            "BERTScore F1":[0.6189, 0.9494]
        },
        "Addestramento 8": {
            "Exact Match": [0.0791, 0.7186], 
            "BLEU-1":      [0.1023, 0.7186], 
            "BERTScore F1":[0.709, 0.9479]
        },
        "Addestramento 9": {
            "Exact Match": [0.0504, 0.7085], 
            "BLEU-1":      [0.0972, 0.7085], 
            "BERTScore F1":[0.703, 0.946]
        }
    }
    #confronta_metriche_tipo_risposta(tipi, dati_tipo_risposta)




    rag_encoder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    mia_domanda = "There are any fractures?"
    # Percorsi presi dalle tue immagini (Usa la 'r' davanti per evitare errori con le barre \)
    path_vqa = r"C:/Users/angel/Downloads/results/chroma_vqa_index"
    path_kb  = r"C:/Users/angel/Downloads/results/chroma_kb_index"

    client_vqa = chromadb.PersistentClient(path=path_vqa)
    print("Collezioni in VQA:", client_vqa.list_collections())

    client_kb = chromadb.PersistentClient(path=path_kb)
    print("Collezioni in KB:", client_kb.list_collections())

    rag_collection = client_vqa.get_collection("vqa_rad_pairs")
    kb_collection = client_kb.get_collection("medical_kb")

    visualize_query_3d(
    collection=rag_collection, 
    encoder=rag_encoder, 
    query_text=mia_domanda, 
    title="La mia domanda nell'Universo VQA"
    )