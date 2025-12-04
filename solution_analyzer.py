import numpy as np
import matplotlib.pyplot as plt
import pygmo as pg
from glob import glob
import os

# ==== FUNZIONI PER ANALIZZARE LE SOLUZIONI ====

def load_best_solution(directory=".", pattern="quantcomm_*.npz"):
    """Carica il file con l'hypervolume migliore"""
    files = glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"Nessun file trovato con pattern {pattern}")
        return None, None, None
    
    # Estrai hypervolume dai nomi dei file
    best_hv = 0
    best_file = None
    
    for f in files:
        try:
            # Formato: quantcomm_X_Y_HV.npz
            hv = int(f.split('_')[-1].replace('.npz', ''))
            if hv > best_hv:
                best_hv = hv
                best_file = f
        except:
            continue
    
    if best_file:
        print(f"Miglior file: {best_file}")
        print(f"Hypervolume: {best_hv / 10000000:.6f}")
        
        with np.load(best_file) as data:
            xs = data['xs']
            ys = data['ys']
        
        return xs, ys, best_file
    
    return None, None, None


def filter_valid_solutions(xs, ys, udp):
    """Filtra solo le soluzioni che rispettano i vincoli"""
    valid_indices = []
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        # Ricalcola fitness completa per avere i vincoli
        x_int = x.copy()
        x_int[10:] = x_int[10:].astype(int)
        full_fitness = udp.fitness(x_int)
        
        # Verifica vincoli (c1 e c2 devono essere <= 0)
        if full_fitness[2] <= 0 and full_fitness[3] <= 0:
            valid_indices.append(i)
    
    return xs[valid_indices], ys[valid_indices]


def decode_solution(x):
    """Decodifica una soluzione nei suoi parametri"""
    params = {
        # Walker 1
        'W1_a': x[0],  # semi-asse maggiore (normalizzato)
        'W1_e': x[1],  # eccentricità
        'W1_i': x[2],  # inclinazione (rad)
        'W1_w': x[3],  # argomento del perigeo (rad)
        'W1_eta': x[4],  # qualità satelliti
        
        # Walker 2
        'W2_a': x[5],
        'W2_e': x[6],
        'W2_i': x[7],
        'W2_w': x[8],
        'W2_eta': x[9],
        
        # Configurazione costellazioni
        'W1_S': int(x[10]),  # satelliti per piano
        'W1_P': int(x[11]),  # numero piani
        'W1_F': int(x[12]),  # phasing
        'W2_S': int(x[13]),
        'W2_P': int(x[14]),
        'W2_F': int(x[15]),
        
        # Rovers
        'rover_indices': [int(x[16]), int(x[17]), int(x[18]), int(x[19])]
    }
    
    # Calcola numero totale satelliti
    params['W1_total'] = params['W1_S'] * params['W1_P']
    params['W2_total'] = params['W2_S'] * params['W2_P']
    params['total_sats'] = params['W1_total'] + params['W2_total']
    
    return params


def print_solution_details(x, y, udp):
    """Stampa i dettagli di una soluzione"""
    params = decode_solution(x)
    
    print("\n" + "="*70)
    print("DETTAGLI SOLUZIONE")
    print("="*70)
    
    print(f"\nOBIETTIVI:")
    print(f"  J1 (Costo comunicazioni): {y[0]:.6f}")
    print(f"  J2 (Costo infrastruttura): {y[1]:.6f}")
    
    print(f"\nWALKER CONSTELLATION 1:")
    print(f"  Semi-asse maggiore: {params['W1_a']:.4f} (× {6371} km)")
    print(f"  Eccentricità: {params['W1_e']:.4f}")
    print(f"  Inclinazione: {np.degrees(params['W1_i']):.2f}°")
    print(f"  Arg. perigeo: {np.degrees(params['W1_w']):.2f}°")
    print(f"  Qualità (eta): {params['W1_eta']:.2f}")
    print(f"  Configurazione: {params['W1_S']}×{params['W1_P']}×{params['W1_F']}")
    print(f"  Totale satelliti: {params['W1_total']}")
    
    print(f"\nWALKER CONSTELLATION 2:")
    print(f"  Semi-asse maggiore: {params['W2_a']:.4f} (× {6371} km)")
    print(f"  Eccentricità: {params['W2_e']:.4f}")
    print(f"  Inclinazione: {np.degrees(params['W2_i']):.2f}°")
    print(f"  Arg. perigeo: {np.degrees(params['W2_w']):.2f}°")
    print(f"  Qualità (eta): {params['W2_eta']:.2f}")
    print(f"  Configurazione: {params['W2_S']}×{params['W2_P']}×{params['W2_F']}")
    print(f"  Totale satelliti: {params['W2_total']}")
    
    print(f"\nTOTALE SATELLITI: {params['total_sats']}")
    print(f"ROVER INDICES: {params['rover_indices']}")
    
    # Verifica vincoli
    x_int = x.copy()
    x_int[10:] = x_int[10:].astype(int)
    full_fitness = udp.fitness(x_int)
    
    print(f"\nVINCOLI:")
    print(f"  Distanza min rovers: {full_fitness[2]:.2f} {'✓ OK' if full_fitness[2] <= 0 else '✗ VIOLATO'}")
    print(f"  Distanza min satelliti: {full_fitness[3]:.2f} {'✓ OK' if full_fitness[3] <= 0 else '✗ VIOLATO'}")
    
    return params


def plot_pareto_front(ys, highlight_indices=None, save_path=None):
    """Visualizza il fronte di Pareto"""
    ref_point = np.array([1.2, 1.4])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Punti del fronte di Pareto
    ax.scatter(ys[:, 0], ys[:, 1], s=50, alpha=0.6, label='Soluzioni valide')
    
    # Evidenzia soluzioni specifiche
    if highlight_indices is not None:
        for idx in highlight_indices:
            ax.scatter(ys[idx, 0], ys[idx, 1], s=200, marker='D', 
                      edgecolors='red', linewidths=2, alpha=0.8,
                      label=f'Soluzione {idx}')
    
    # Reference point
    ax.scatter(ref_point[0], ref_point[1], s=200, marker='X', 
              c='black', label='Reference point', zorder=10)
    
    # Calcola hypervolume
    try:
        hv = pg.hypervolume(ys)
        hv_value = hv.compute(ref_point)
        ax.text(0.95, 0.95, f'Hypervolume: {hv_value:.6f}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=12)
    except:
        pass
    
    ax.set_xlabel('Cost of communications (J1)', fontsize=12)
    ax.set_ylabel('Cost of infrastructure (J2)', fontsize=12)
    ax.set_title('Pareto Front - Quantum Communications Constellation', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def find_best_solutions(xs, ys):
    """Trova le soluzioni migliori secondo diversi criteri"""
    results = {}
    
    # Miglior J1 (comunicazioni)
    idx_j1 = np.argmin(ys[:, 0])
    results['best_j1'] = {
        'index': idx_j1,
        'x': xs[idx_j1],
        'y': ys[idx_j1],
        'description': 'Miglior costo comunicazioni'
    }
    
    # Miglior J2 (infrastruttura)
    idx_j2 = np.argmin(ys[:, 1])
    results['best_j2'] = {
        'index': idx_j2,
        'x': xs[idx_j2],
        'y': ys[idx_j2],
        'description': 'Miglior costo infrastruttura'
    }
    
    # Soluzione bilanciata (minima distanza euclidea dall'origine)
    distances = np.linalg.norm(ys, axis=1)
    idx_balanced = np.argmin(distances)
    results['balanced'] = {
        'index': idx_balanced,
        'x': xs[idx_balanced],
        'y': ys[idx_balanced],
        'description': 'Soluzione bilanciata'
    }
    
    return results


# ==== SCRIPT PRINCIPALE ====

if __name__ == '__main__':
    # Importa UDP
    from constellation_udp import constellation_udp  # Assumendo che sia nel tuo script
    
    print("Caricamento soluzioni...")
    xs, ys, filename = load_best_solution()
    
    if xs is None:
        print("Nessuna soluzione trovata!")
        exit()
    
    # Crea UDP per validazione
    udp = constellation_udp()
    
    print(f"\nTrovate {len(xs)} soluzioni nel fronte di Pareto")
    
    # Trova le soluzioni migliori
    print("\n" + "="*70)
    print("RICERCA SOLUZIONI OTTIMALI")
    print("="*70)
    
    best_solutions = find_best_solutions(xs, ys)
    
    # Stampa dettagli delle soluzioni chiave
    for key, sol in best_solutions.items():
        print(f"\n{sol['description'].upper()}:")
        print_solution_details(sol['x'], sol['y'], udp)
    
    # Visualizza il fronte di Pareto
    print("\nGenerazione grafico Pareto Front...")
    highlight_indices = [sol['index'] for sol in best_solutions.values()]
    plot_pareto_front(ys, highlight_indices=highlight_indices, 
                     save_path='pareto_front_analysis.png')
    
    # Salva report dettagliato
    report_path = 'solutions_report.txt'
    with open(report_path, 'w') as f:
        f.write("QUANTUM COMMUNICATIONS CONSTELLATION - REPORT SOLUZIONI\n")
        f.write("="*70 + "\n\n")
        f.write(f"File sorgente: {filename}\n")
        f.write(f"Numero soluzioni: {len(xs)}\n\n")
        
        for key, sol in best_solutions.items():
            f.write(f"\n{sol['description'].upper()}\n")
            f.write("-"*70 + "\n")
            params = decode_solution(sol['x'])
            f.write(f"J1: {sol['y'][0]:.6f}, J2: {sol['y'][1]:.6f}\n")
            f.write(f"Satelliti totali: {params['total_sats']}\n")
            f.write(f"Walker 1: {params['W1_total']} sat, eta={params['W1_eta']:.2f}\n")
            f.write(f"Walker 2: {params['W2_total']} sat, eta={params['W2_eta']:.2f}\n\n")
    
    print(f"\nReport salvato in: {report_path}")
