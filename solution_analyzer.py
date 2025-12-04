import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ============================================
# CARICA LE SOLUZIONI DAL FILE .npz
# ============================================

def load_solutions(filename):
    """Carica le soluzioni dal file .npz"""
    with np.load(filename) as data:
        xs = data['xs']  # Parametri orbitali
        ys = data['ys']  # Obiettivi (f1, f2, [c1, c2])
    return xs, ys

# ============================================
# ANALIZZA I PARAMETRI ORBITALI
# ============================================

def decode_solution(x):
    """Decodifica una soluzione in parametri leggibili"""
    params = {
        # Walker Constellation 1
        'W1_semiasse_maggiore_km': x[0] * 6371,  # a1 normalizzato
        'W1_eccentricita': x[1],
        'W1_inclinazione_deg': np.degrees(x[2]),
        'W1_arg_perigeo_deg': np.degrees(x[3]),
        'W1_qualita_eta': x[4],
        
        # Walker Constellation 2
        'W2_semiasse_maggiore_km': x[5] * 6371,  # a2 normalizzato
        'W2_eccentricita': x[6],
        'W2_inclinazione_deg': np.degrees(x[7]),
        'W2_arg_perigeo_deg': np.degrees(x[8]),
        'W2_qualita_eta': x[9],
        
        # Configurazione Walker 1
        'W1_sat_per_piano': int(x[10]),
        'W1_num_piani': int(x[11]),
        'W1_phasing': int(x[12]),
        'W1_tot_satelliti': int(x[10]) * int(x[11]),
        
        # Configurazione Walker 2
        'W2_sat_per_piano': int(x[13]),
        'W2_num_piani': int(x[14]),
        'W2_phasing': int(x[15]),
        'W2_tot_satelliti': int(x[13]) * int(x[14]),
        
        # Rovers
        'rover_indices': [int(x[16]), int(x[17]), int(x[18]), int(x[19])],
        
        # Totali
        'satelliti_totali': int(x[10])*int(x[11]) + int(x[13])*int(x[14])
    }
    return params

# ============================================
# CALCOLA L'IPERVOLUME
# ============================================

def calculate_hypervolume(ys, ref_point=[1.2, 1.4]):
    """Calcola l'ipervolume del fronte di Pareto"""
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    
    # Filtra punti dominati dal reference point
    valid_points = (f1 <= ref_point[0]) & (f2 <= ref_point[1])
    
    if not np.any(valid_points):
        return 0.0
    
    # Ordina per f1
    sorted_indices = np.argsort(f1[valid_points])
    sorted_f1 = f1[valid_points][sorted_indices]
    sorted_f2 = f2[valid_points][sorted_indices]
    
    # Calcola ipervolume
    hypervolume = 0.0
    prev_f1 = 0.0
    
    for i in range(len(sorted_f1)):
        width = sorted_f1[i] - prev_f1
        height = ref_point[1] - sorted_f2[i]
        hypervolume += width * height
        prev_f1 = sorted_f1[i]
    
    # Aggiungi ultimo rettangolo
    width = ref_point[0] - prev_f1
    height = ref_point[1] - sorted_f2[-1]
    hypervolume += width * height
    
    return hypervolume

# ============================================
# VISUALIZZA IL FRONTE DI PARETO
# ============================================

def plot_pareto_front(ys, best_solutions, hypervolume, filename='pareto_front.png'):
    """Visualizza il fronte di Pareto con soluzioni notevoli evidenziate"""
    plt.figure(figsize=(12, 7))
    
    # Estrai solo f1 e f2 (ignora eventuali vincoli)
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    
    # Disegna tutte le soluzioni
    plt.scatter(f1, f2, alpha=0.4, s=50, c='lightblue', edgecolors='navy', 
                linewidth=0.5, label='Fronte di Pareto')
    
    # Evidenzia le soluzioni notevoli
    colors = {'Migliore Comunicazione': 'green', 
              'Minor Costo': 'red', 
              'Bilanciata': 'orange'}
    
    for name, (x, y) in best_solutions.items():
        plt.scatter(y[0], y[1], s=300, c=colors[name], marker='*', 
                   edgecolors='black', linewidth=2, 
                   label=f'{name}\n(f1={y[0]:.4f}, f2={y[1]:.4f})', 
                   zorder=5)
    
    plt.xlabel('f1: Costo Comunicazione (normalizzato)', fontsize=12)
    plt.ylabel('f2: Costo Infrastruttura (normalizzato)', fontsize=12)
    plt.title(f'Fronte di Pareto - Ipervolume: {hypervolume:.6f}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Aggiungi punto di riferimento
    plt.axvline(x=1.2, color='purple', linestyle='--', alpha=0.3, linewidth=2, label='Reference Point')
    plt.axhline(y=1.4, color='purple', linestyle='--', alpha=0.3, linewidth=2)
    
    plt.legend(loc='best', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Grafico salvato in: {filename}")

# ============================================
# TROVA LE SOLUZIONI "MIGLIORI"
# ============================================

def find_best_solutions(xs, ys):
    """Identifica soluzioni notevoli nel fronte di Pareto"""
    
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    
    # 1. Soluzione con miglior comunicazione (minimo f1)
    idx_best_comm = np.argmin(f1)
    
    # 2. Soluzione con minor costo (minimo f2)
    idx_best_cost = np.argmin(f2)
    
    # 3. Soluzione "bilanciata" (minima distanza euclidea dall'origine)
    distances = np.sqrt(f1**2 + f2**2)
    idx_balanced = np.argmin(distances)
    
    solutions = {
        'Migliore Comunicazione': (xs[idx_best_comm], ys[idx_best_comm]),
        'Minor Costo': (xs[idx_best_cost], ys[idx_best_cost]),
        'Bilanciata': (xs[idx_balanced], ys[idx_balanced])
    }
    
    return solutions

# ============================================
# STAMPA DETTAGLI COMPLETI SOLUZIONE
# ============================================

def print_solution_details(name, x, y):
    """Stampa tutti i parametri di una soluzione"""
    params = decode_solution(x)
    
    print(f"\n{'='*70}")
    print(f"▶ {name.upper()}")
    print(f"{'='*70}")
    print(f"Obiettivi: f1={y[0]:.6f}, f2={y[1]:.6f}")
    print(f"Satelliti totali: {params['satelliti_totali']}")
    
    print(f"\n--- WALKER CONSTELLATION 1 ---")
    print(f"  Satelliti totali: {params['W1_tot_satelliti']} ({params['W1_sat_per_piano']}×{params['W1_num_piani']})")
    print(f"  Qualità η: {params['W1_qualita_eta']:.3f}")
    print(f"  Semiasse maggiore: {params['W1_semiasse_maggiore_km']:.2f} km")
    print(f"  Eccentricità: {params['W1_eccentricita']:.6f}")
    print(f"  Inclinazione: {params['W1_inclinazione_deg']:.2f}°")
    print(f"  Argomento perigeo: {params['W1_arg_perigeo_deg']:.2f}°")
    print(f"  Phasing: {params['W1_phasing']}")
    
    print(f"\n--- WALKER CONSTELLATION 2 ---")
    print(f"  Satelliti totali: {params['W2_tot_satelliti']} ({params['W2_sat_per_piano']}×{params['W2_num_piani']})")
    print(f"  Qualità η: {params['W2_qualita_eta']:.3f}")
    print(f"  Semiasse maggiore: {params['W2_semiasse_maggiore_km']:.2f} km")
    print(f"  Eccentricità: {params['W2_eccentricita']:.6f}")
    print(f"  Inclinazione: {params['W2_inclinazione_deg']:.2f}°")
    print(f"  Argomento perigeo: {params['W2_arg_perigeo_deg']:.2f}°")
    print(f"  Phasing: {params['W2_phasing']}")
    
    print(f"\n--- ROVERS ---")
    print(f"  Indices: {params['rover_indices']}")

# ============================================
# ESPORTA IN CSV PER ANALISI
# ============================================

def export_to_csv(xs, ys, filename='soluzioni_pareto.csv'):
    """Esporta tutte le soluzioni in CSV"""
    
    data = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        params = decode_solution(x)
        row = {
            'ID': i,
            'f1_comunicazione': y[0],
            'f2_costo': y[1],
            **params
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"CSV esportato in: {filename}")
    return df

# ============================================
# ESEMPIO DI UTILIZZO
# ============================================

if __name__ == '__main__':
    
    # CARICA IL FILE (sostituisci con il tuo file .npz)
    filename = 'quantcomm_1_100_6372134.npz'  # O il file più recente generato
    
    try:
        xs, ys = load_solutions(filename)
        print(f"✓ Caricate {len(xs)} soluzioni dal file {filename}")
        print(f"  - Dimensione xs: {xs.shape}")
        print(f"  - Dimensione ys: {ys.shape}\n")
        
        # TROVA SOLUZIONI NOTEVOLI
        best_sols = find_best_solutions(xs, ys)
        
        # CALCOLA IPERVOLUME
        hypervolume = calculate_hypervolume(ys)
        print(f"\n{'='*70}")
        print(f"IPERVOLUME DEL FRONTE DI PARETO: {hypervolume:.6f}")
        print(f"{'='*70}")
        
        # VISUALIZZA IL FRONTE DI PARETO
        plot_pareto_front(ys, best_sols, hypervolume)
        
        # STAMPA DETTAGLI COMPLETI DELLE SOLUZIONI NOTEVOLI
        for name, (x, y) in best_sols.items():
            print_solution_details(name, x, y)
        
        # ESPORTA TUTTO IN CSV
        print("\n" + "="*70)
        df = export_to_csv(xs, ys)
        print(f"\nPrime 5 soluzioni:")
        print(df[['ID', 'f1_comunicazione', 'f2_costo', 'satelliti_totali']].head())
        
    except FileNotFoundError:
        print(f"❌ File {filename} non trovato!")
        print("   Assicurati che l'ottimizzazione abbia generato il file .npz")
        print("   I file vengono salvati nella directory corrente con nomi tipo:")
        print("   - quantcomm_6372134.npz")
        print("   - quantcomm_1_100_6372134.npz")
