import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Nomi dei file di testo caricati
files = [
    "addestramento.txt", "addestramento2.txt", "addestramento3.txt",
    "addestramento4.txt", "addestramento5.txt", "Addestramento6.txt",
    "Addestramento7.txt", "Addestramento8.txt", "Addestramento9.txt",
    "AddestramentoFinale.txt"
]

def parse_files(file_list):
    """
    Legge tutti i file txt e sfrutta le espressioni regolari (Regex) 
    per estrarre Validation Loss per ogni epoca, e le 3 metriche finali.
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
    Funzione 2: Bar Plot che confronta le 3 metriche (Exact Match, BLEU, BERTScore) 
    a livello di inferenza
    """
    labels = []
    exact_matches = []
    bleus = []
    bert_scores = []
    
    for filename, metrics in data.items():
        # Scartiamo i file che non hanno risultati di inferenza, come l'addestramento 2
        if metrics['exact_match'] > 0 or metrics['bleu1'] > 0 or metrics['bert_f1'] > 0:
            labels.append(filename.replace('.txt', '').replace('addestramento', 'Add. '))
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

# --- Esecuzione del Codice ---
if __name__ == "__main__":
    dati_estratti = parse_files(files)
    plot_inference_metrics(dati_estratti)