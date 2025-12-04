import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import sys

def is_dominated(point, other_points):
    """Verifica se un punto è dominato da altri punti"""
    for other in other_points:
        if all(other <= point) and any(other < point):
            return True
    return False

def find_pareto_front(objectives):
    """Trova il fronte di Pareto da un set di obiettivi"""
    pareto_mask = np.ones(len(objectives), dtype=bool)
    
    for i, point in enumerate(objectives):
        if pareto_mask[i]:
            # Rimuovi dalla maschera tutti i punti dominati da questo
            other_points = objectives[pareto_mask]
            other_indices = np.where(pareto_mask)[0]
            
            for j, other in zip(other_indices, other_points):
                if i != j:
                    if all(point <= other) and any(point < other):
                        pareto_mask[j] = False
    
    return pareto_mask

def plot_solutions_and_pareto(file_path, dark_mode=True):
    """
    Visualizza tutte le soluzioni dal file .npz e evidenzia il fronte di Pareto
    
    Args:
        file_path: path al file .npz
        dark_mode: usa sfondo scuro
    """
    
    # Carica il file
    with np.load(file_path) as data:
        xs = data['xs']
        ys = data['ys']
    
    # Estrai gli obiettivi (prime 2 colonne)
    objectives = ys[:, :2] if ys.shape[1] > 2 else ys
    
    # Trova il fronte di Pareto
    pareto_mask = find_pareto_front(objectives)
    pareto_points = objectives[pareto_mask]
    non_pareto_points = objectives[~pareto_mask]
    
    # Setup plot
    if dark_mode:
        plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot soluzioni non-Pareto (in grigio/trasparente)
    if len(non_pareto_points) > 0:
        ax.scatter(non_pareto_points[:, 0], non_pareto_points[:, 1], 
                  c='gray', s=50, alpha=0.3, label='Soluzioni dominate', 
                  edgecolors='none')
    
    # Plot fronte di Pareto (evidenziato)
    ax.scatter(pareto_points[:, 0], pareto_points[:, 1], 
              c='cyan', s=100, alpha=0.8, label='Fronte di Pareto',
              edgecolors='white', linewidth=1.5, zorder=5)
    
    # Connetti i punti del fronte di Pareto
    sorted_indices = np.argsort(pareto_points[:, 0])
    sorted_pareto = pareto_points[sorted_indices]
    ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 
           'c--', alpha=0.5, linewidth=2, zorder=4)
    
    # Reference point
    ref_point = np.array([1.2, 1.4])
    ax.scatter(ref_point[0], ref_point[1], 
              c='red', s=300, marker='*', 
              label='Reference Point', zorder=6, 
              edgecolors='yellow', linewidth=2)
    
    # Linee tratteggiate dal reference point
    ax.axvline(x=ref_point[0], color='red', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(y=ref_point[1], color='red', linestyle=':', alpha=0.3, linewidth=1)
    
    # Labels e titolo
    ax.set_xlabel('Obiettivo 1: Costo Comunicazione Medio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Obiettivo 2: Costo Infrastruttura', fontsize=14, fontweight='bold')
    
    file_name = Path(file_path).stem
    ax.set_title(f'Soluzioni e Fronte di Pareto - {file_name}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Statistiche nel plot
    stats_text = f'Soluzioni totali: {len(objectives)}\n'
    stats_text += f'Fronte di Pareto: {len(pareto_points)}\n'
    stats_text += f'Dominate: {len(non_pareto_points)}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
           color='white')
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Salva
    output_name = f"{file_name}_pareto_front.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato: {output_name}")
    
    plt.show()
    
    # Stampa statistiche
    print("\n" + "="*60)
    print(f"ANALISI FILE: {file_name}")
    print("="*60)
    print(f"Soluzioni totali: {len(objectives)}")
    print(f"Soluzioni sul fronte di Pareto: {len(pareto_points)} ({len(pareto_points)/len(objectives)*100:.1f}%)")
    print(f"Soluzioni dominate: {len(non_pareto_points)} ({len(non_pareto_points)/len(objectives)*100:.1f}%)")
    print("\nFronte di Pareto - Range obiettivi:")
    print(f"  Obiettivo 1: [{pareto_points[:, 0].min():.4f}, {pareto_points[:, 0].max():.4f}]")
    print(f"  Obiettivo 2: [{pareto_points[:, 1].min():.4f}, {pareto_points[:, 1].max():.4f}]")
    
    # Calcola hypervolume se possibile
    try:
        import pygmo as pg
        valid = [obj for obj in pareto_points if all(obj <= ref_point)]
        if len(valid) > 0:
            hv = pg.hypervolume(valid)
            hv_value = hv.compute(ref_point) * 10000
            print(f"\nHypervolume: {hv_value:.2f}")
    except ImportError:
        pass
    
    print("="*60)
    
    return fig, ax, pareto_points, non_pareto_points


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python script.py <file.npz>")
        print("Esempio: python script.py quantcomm_639739.npz")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"Errore: file '{file_path}' non trovato")
        sys.exit(1)
    
    plot_solutions_and_pareto(file_path)
