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
    
    # Filtra solo i punti dominati dal reference point
    valid_mask = (f1 < ref_point[0]) & (f2 < ref_point[1])
    f1_valid = f1[valid_mask]
    f2_valid = f2[valid_mask]
    
    if len(f1_valid) == 0:
        return 0.0
    
    # Ordina per f1
    sorted_indices = np.argsort(f1_valid)
    f1_sorted = f1_valid[sorted_indices]
    f2_sorted = f2_valid[sorted_indices]
    
    # Calcola l'ipervolume
    hypervolume = 0.0
    prev_f1 = 0.0
    
    for i in range(len(f1_sorted)):
        width = f1_sorted[i] - prev_f1
        height = ref_point[1] - f2_sorted[i]
        hypervolume += width * height
        prev_f1 = f1_sorted[i]
    
    # Aggiungi l'ultimo rettangolo
    width = ref_point[0] - prev_f1
    height = ref_point[1] - f2_sorted[-1]
    hypervolume += width * height
    
    return hypervolume

# ============================================
# VISUALIZZA IL FRONTE DI PARETO
# ============================================

def plot_pareto_front(ys, xs=None, filename='pareto_front.png'):
    """Visualizza il fronte di Pareto con soluzioni notevoli evidenziate"""
    plt.figure(figsize=(12, 8))
    
    # Estrai solo f1 e f2 (ignora eventuali vincoli)
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    
    # Plot di tutte le soluzioni
    plt.scatter(f1, f2, alpha=0.6, s=50, color='#1f77b4', label='Soluzioni Pareto')
    
    # Se xs è fornito, evidenzia le soluzioni notevoli
    if xs is not None:
        # Trova le soluzioni notevoli
        idx_best_comm = np.argmin(f1)
        idx_best_cost = np.argmin(f2)
        distances = np.sqrt(f1**2 + f2**2)
        idx_balanced = np.argmin(distances)
        
        # Plot soluzioni notevoli con colori e simboli diversi
        plt.scatter(f1[idx_best_comm], f2[idx_best_comm], 
                   s=200, color='#ff7f0e', marker='D', 
                   edgecolors='black', linewidth=2,
                   label='A: Migliore Comunicazione', zorder=5)
        
        plt.scatter(f1[idx_best_cost], f2[idx_best_cost], 
                   s=200, color='#d62728', marker='s', 
                   edgecolors='black', linewidth=2,
                   label='B: Minor Costo', zorder=5)
        
        plt.scatter(f1[idx_balanced], f2[idx_balanced], 
                   s=200, color='#2ca02c', marker='o', 
                   edgecolors='black', linewidth=2,
                   label='C: Bilanciata', zorder=5)
        
        # Calcola l'ipervolume
        hypervolume = calculate_hypervolume(ys, ref_point=[1.2, 1.4])
        
        # Aggiungi testo con ipervolume
        plt.text(0.98, 0.98, f'Hypervolume:\n{hypervolume:,.3f}', 
                transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel('Cost of communications between rovers and motherships — $f_1$', fontsize=12)
    plt.ylabel('Cost of building and operating the\ntwo satellite constellations — $f_2$', fontsize=12)
    plt.title('Fronte di Pareto - Soluzioni Ottimali', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Aggiungi punto di riferimento
    plt.axvline(x=1.2, color='r', linestyle='--', alpha=0.3, linewidth=1)
    plt.axhline(y=1.4, color='r', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.legend(loc='upper right', fontsize=10, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
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
        
        # VISUALIZZA IL FRONTE DI PARETO con soluzioni notevoli
        plot_pareto_front(ys, xs)
        
        # TROVA SOLUZIONI NOTEVOLI
        print("\n" + "="*60)
        print("SOLUZIONI NOTEVOLI NEL FRONTE DI PARETO")
        print("="*60)
        
        best_sols = find_best_solutions(xs, ys)
        
        for name, (x, y) in best_sols.items():
            marker = 'A' if 'Comunicazione' in name else ('B' if 'Costo' in name else 'C')
            print(f"\n▶ [{marker}] {name}:")
            print(f"  Obiettivi: f1={y[0]:.6f}, f2={y[1]:.6f}")
            params = decode_solution(x)
            
            print(f"\n  📡 Walker Constellation 1:")
            print(f"     - Satelliti totali: {params['W1_tot_satelliti']}")
            print(f"     - Configurazione: {params['W1_sat_per_piano']}×{params['W1_num_piani']} (phasing: {params['W1_phasing']})")
            print(f"     - Semiasse maggiore: {params['W1_semiasse_maggiore_km']:.2f} km")
            print(f"     - Eccentricità: {params['W1_eccentricita']:.6f}")
            print(f"     - Inclinazione: {params['W1_inclinazione_deg']:.2f}°")
            print(f"     - Argomento perigeo: {params['W1_arg_perigeo_deg']:.2f}°")
            print(f"     - Qualità η: {params['W1_qualita_eta']:.6f}")
            
            print(f"\n  📡 Walker Constellation 2:")
            print(f"     - Satelliti totali: {params['W2_tot_satelliti']}")
            print(f"     - Configurazione: {params['W2_sat_per_piano']}×{params['W2_num_piani']} (phasing: {params['W2_phasing']})")
            print(f"     - Semiasse maggiore: {params['W2_semiasse_maggiore_km']:.2f} km")
            print(f"     - Eccentricità: {params['W2_eccentricita']:.6f}")
            print(f"     - Inclinazione: {params['W2_inclinazione_deg']:.2f}°")
            print(f"     - Argomento perigeo: {params['W2_arg_perigeo_deg']:.2f}°")
            print(f"     - Qualità η: {params['W2_qualita_eta']:.6f}")
            
            print(f"\n  🚗 Rovers: {params['rover_indices']}")
            print(f"  🛰️  Satelliti totali sistema: {params['satelliti_totali']}")
            print("-" * 60)
        
        # ESPORTA TUTTO IN CSV
        print("\n" + "="*60)
        df = export_to_csv(xs, ys)
        print(f"\nPrime 5 soluzioni:")
        print(df[['ID', 'f1_comunicazione', 'f2_costo', 'satelliti_totali']].head())
        
    except FileNotFoundError:
        print(f"❌ File {filename} non trovato!")
        print("   Assicurati che l'ottimizzazione abbia generato il file .npz")
        print("   I file vengono salvati nella directory corrente con nomi tipo:")
        print("   - quantcomm_6372134.npz")
        print("   - quantcomm_1_100_6372134.npz")
