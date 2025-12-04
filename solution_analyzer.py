import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from matplotlib import cm

def load_solution(filepath):
    """Carica una soluzione da file .npz"""
    with np.load(filepath) as data:
        xs = data['xs']
        ys = data['ys']
    return xs, ys

def plot_pareto_fronts(solution_files, title="Fronti di Pareto", 
                       save_path=None, dark_mode=True):
    """
    Visualizza più fronti di Pareto da diversi file di soluzione
    
    Args:
        solution_files: lista di path ai file .npz
        title: titolo del grafico
        save_path: percorso dove salvare il grafico (opzionale)
        dark_mode: usa sfondo scuro
    """
    if dark_mode:
        plt.style.use('dark_background')
        sns.set_palette("bright")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colori per le diverse soluzioni
    colors = cm.rainbow(np.linspace(0, 1, len(solution_files)))
    
    all_ys = []
    
    for idx, file_path in enumerate(solution_files):
        try:
            xs, ys = load_solution(file_path)
            
            # Estrai solo gli obiettivi (prime 2 colonne)
            if ys.shape[1] > 2:
                objectives = ys[:, :2]
            else:
                objectives = ys
            
            all_ys.append(objectives)
            
            # Nome file per la leggenda
            file_name = Path(file_path).stem
            
            # Plot delle soluzioni
            ax.scatter(objectives[:, 0], objectives[:, 1], 
                      c=[colors[idx]], label=file_name, 
                      s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
            
            # Connetti i punti del fronte
            sorted_indices = np.argsort(objectives[:, 0])
            sorted_obj = objectives[sorted_indices]
            ax.plot(sorted_obj[:, 0], sorted_obj[:, 1], 
                   c=colors[idx], alpha=0.3, linewidth=1)
            
        except Exception as e:
            print(f"Errore nel caricamento di {file_path}: {e}")
    
    # Reference point
    ref_point = np.array([1.2, 1.4])
    ax.scatter(ref_point[0], ref_point[1], 
              c='red', s=200, marker='*', 
              label='Reference Point', zorder=5, 
              edgecolors='white', linewidth=2)
    
    # Linee tratteggiate dal reference point
    ax.axvline(x=ref_point[0], color='red', linestyle='--', alpha=0.3)
    ax.axhline(y=ref_point[1], color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Obiettivo 1: Costo Comunicazione Medio', fontsize=12)
    ax.set_ylabel('Obiettivo 2: Costo Infrastruttura', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")
    
    plt.show()
    
    return fig, ax, all_ys

def plot_single_pareto(file_path, title=None, save_path=None, dark_mode=True):
    """Visualizza un singolo fronte di Pareto con dettagli"""
    
    if dark_mode:
        plt.style.use('dark_background')
    
    xs, ys = load_solution(file_path)
    
    # Estrai obiettivi e vincoli
    objectives = ys[:, :2]
    
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Fronte di Pareto
    ax1 = plt.subplot(1, 3, 1)
    scatter = ax1.scatter(objectives[:, 0], objectives[:, 1], 
                         c=range(len(objectives)), cmap='viridis',
                         s=100, alpha=0.6, edgecolors='white', linewidth=1)
    
    # Connetti i punti
    sorted_indices = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_indices]
    ax1.plot(sorted_obj[:, 0], sorted_obj[:, 1], 
            'w--', alpha=0.3, linewidth=1)
    
    # Reference point
    ref_point = np.array([1.2, 1.4])
    ax1.scatter(ref_point[0], ref_point[1], 
               c='red', s=200, marker='*', label='Reference Point')
    
    ax1.set_xlabel('Obiettivo 1: Costo Comunicazione')
    ax1.set_ylabel('Obiettivo 2: Costo Infrastruttura')
    ax1.set_title('Fronte di Pareto')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Indice Soluzione')
    
    # Subplot 2: Distribuzione Obiettivo 1
    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(objectives[:, 0], bins=20, alpha=0.7, color='cyan', edgecolor='white')
    ax2.axvline(ref_point[0], color='red', linestyle='--', 
               label=f'Ref: {ref_point[0]:.2f}')
    ax2.set_xlabel('Obiettivo 1')
    ax2.set_ylabel('Frequenza')
    ax2.set_title('Distribuzione Costo Comunicazione')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Distribuzione Obiettivo 2
    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(objectives[:, 1], bins=20, alpha=0.7, color='magenta', edgecolor='white')
    ax3.axvline(ref_point[1], color='red', linestyle='--', 
               label=f'Ref: {ref_point[1]:.2f}')
    ax3.set_xlabel('Obiettivo 2')
    ax3.set_ylabel('Frequenza')
    ax3.set_title('Distribuzione Costo Infrastruttura')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")
    
    plt.show()
    
    return fig

def compute_hypervolume_evolution(solution_files, ref_point=[1.2, 1.4]):
    """Calcola e visualizza l'evoluzione dell'hypervolume"""
    try:
        import pygmo as pg
    except ImportError:
        print("Pygmo non disponibile per il calcolo dell'hypervolume")
        return None
    
    plt.style.use('dark_background')
    
    hypervolumes = []
    iterations = []
    
    for file_path in solution_files:
        try:
            xs, ys = load_solution(file_path)
            objectives = ys[:, :2] if ys.shape[1] > 2 else ys
            
            # Filtra solo soluzioni valide che dominano il reference point
            valid = [obj for obj in objectives if all(obj <= ref_point)]
            
            if len(valid) > 0:
                hv = pg.hypervolume(valid)
                hypervolumes.append(hv.compute(ref_point) * 10000)
                
                # Estrai numero iterazione dal nome file
                file_name = Path(file_path).stem
                parts = file_name.split('_')
                if len(parts) > 1 and parts[1].isdigit():
                    iterations.append(int(parts[1]))
                else:
                    iterations.append(len(iterations))
            
        except Exception as e:
            print(f"Errore nel calcolo HV per {file_path}: {e}")
    
    if hypervolumes:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, hypervolumes, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Iterazione', fontsize=12)
        ax.set_ylabel('Hypervolume × 10000', fontsize=12)
        ax.set_title('Evoluzione Hypervolume', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fig, iterations, hypervolumes
    
    return None

# Esempio d'uso
if __name__ == "__main__":
    # Lista i file .npz nella directory corrente
    solution_files = list(Path('.').glob('quantcomm*.npz'))
    
    if not solution_files:
        print("Nessun file .npz trovato nella directory corrente")
        print("Assicurati di essere nella directory dove sono salvate le soluzioni")
    else:
        print(f"Trovati {len(solution_files)} file di soluzione")
        
        # Ordina per nome
        solution_files = sorted(solution_files)
        
        # Visualizza tutti i fronti insieme
        if len(solution_files) > 1:
            plot_pareto_fronts(solution_files[:5],  # mostra max 5 per leggibilità
                              title="Confronto Fronti di Pareto",
                              save_path="pareto_comparison.png")
        
        # Visualizza dettagli dell'ultima soluzione
        if solution_files:
            plot_single_pareto(solution_files[-1], 
                             title=f"Analisi Dettagliata: {solution_files[-1].stem}",
                             save_path="pareto_detail.png")
        
        # Evoluzione hypervolume
        if len(solution_files) > 1:
            compute_hypervolume_evolution(solution_files)
