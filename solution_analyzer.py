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
# VISUALIZZA IL FRONTE DI PARETO
# ============================================

def plot_pareto_front(ys, filename='pareto_front.png'):
    """Visualizza il fronte di Pareto"""
    plt.figure(figsize=(10, 6))
    
    # Estrai solo f1 e f2 (ignora eventuali vincoli)
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    
    plt.scatter(f1, f2, alpha=0.6, s=50)
    plt.xlabel('f1: Costo Comunicazione (normalizzato)', fontsize=12)
    plt.ylabel('f2: Costo Infrastruttura (normalizzato)', fontsize=12)
    plt.title('Fronte di Pareto - Soluzioni Ottimali', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Aggiungi punto di riferimento
    plt.axvline(x=1.2, color='r', linestyle='--', alpha=0.3, label='Ref point')
    plt.axhline(y=1.4, color='r', linestyle='--', alpha=0.3)
    
    plt.legend()
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
        
        # VISUALIZZA IL FRONTE DI PARETO
        plot_pareto_front(ys)
        
        # TROVA SOLUZIONI NOTEVOLI
        print("\n" + "="*60)
        print("SOLUZIONI NOTEVOLI NEL FRONTE DI PARETO")
        print("="*60)
        
        best_sols = find_best_solutions(xs, ys)
        
        for name, (x, y) in best_sols.items():
            print(f"\n▶ {name}:")
            print(f"  Obiettivi: f1={y[0]:.4f}, f2={y[1]:.4f}")
            params = decode_solution(x)
            print(f"  Satelliti totali: {params['satelliti_totali']}")
            print(f"  W1: {params['W1_tot_satelliti']} sat (η={params['W1_qualita_eta']:.1f})")
            print(f"  W2: {params['W2_tot_satelliti']} sat (η={params['W2_qualita_eta']:.1f})")
        
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
