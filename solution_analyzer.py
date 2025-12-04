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
# VERIFICA VINCOLI E DOMINANZA
# ============================================

def check_constraints(ys):
    """
    Verifica se le soluzioni rispettano i vincoli.
    Assume che ys possa avere 2 colonne (solo obiettivi) o 4 colonne (obiettivi + vincoli)
    """
    if ys.shape[1] == 2:
        # Nessun vincolo presente, tutte le soluzioni sono valide
        return np.ones(len(ys), dtype=bool)
    elif ys.shape[1] >= 4:
        # Vincoli presenti nelle colonne 2 e 3 (c1, c2)
        # Una soluzione è valida se tutti i vincoli <= 0
        c1 = ys[:, 2]
        c2 = ys[:, 3]
        return (c1 <= 0) & (c2 <= 0)
    else:
        return np.ones(len(ys), dtype=bool)

def is_pareto_efficient(costs):
    """
    Identifica le soluzioni appartenenti al fronte di Pareto.
    Una soluzione è non-dominata se nessun'altra soluzione è migliore in tutti gli obiettivi.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Trova soluzioni che dominano la soluzione i
            # Una soluzione j domina i se è migliore o uguale in tutti gli obiettivi
            # e strettamente migliore in almeno uno
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

def classify_solutions(ys):
    """
    Classifica le soluzioni in:
    - Pareto valide (non dominate e rispettano vincoli)
    - Pareto non valide (non dominate ma violano vincoli)
    - Dominate valide (dominate ma rispettano vincoli)
    - Dominate non valide (dominate e violano vincoli)
    """
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    costs = np.column_stack([f1, f2])
    
    # Verifica vincoli
    valid = check_constraints(ys)
    
    # Verifica dominanza
    pareto = is_pareto_efficient(costs)
    
    # Classifica
    pareto_valid = pareto & valid
    pareto_invalid = pareto & ~valid
    dominated_valid = ~pareto & valid
    dominated_invalid = ~pareto & ~valid
    
    return {
        'pareto_valid': pareto_valid,
        'pareto_invalid': pareto_invalid,
        'dominated_valid': dominated_valid,
        'dominated_invalid': dominated_invalid,
        'valid': valid,
        'pareto': pareto
    }

# ============================================
# CALCOLA L'IPERVOLUME
# ============================================

def calculate_hypervolume(ys, ref_point=[1.2, 1.4]):
    """Calcola l'ipervolume del fronte di Pareto (solo soluzioni valide e non dominate)"""
    classification = classify_solutions(ys)
    pareto_valid_mask = classification['pareto_valid']
    
    if not np.any(pareto_valid_mask):
        return 0.0
    
    f1 = ys[pareto_valid_mask, 0]
    f2 = ys[pareto_valid_mask, 1]
    
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
    """Visualizza il fronte di Pareto con classificazione delle soluzioni"""
    plt.figure(figsize=(12, 8))
    
    # Estrai obiettivi
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    
    # Classifica le soluzioni
    classification = classify_solutions(ys)
    
    # Conta le soluzioni per categoria
    n_pareto_valid = np.sum(classification['pareto_valid'])
    n_pareto_invalid = np.sum(classification['pareto_invalid'])
    n_dominated_valid = np.sum(classification['dominated_valid'])
    n_dominated_invalid = np.sum(classification['dominated_invalid'])
    
    # Plot delle diverse categorie con ordine di z-index appropriato
    # Dominate per ultime (sotto), Pareto per prime (sopra)
    
    if n_dominated_invalid > 0:
        plt.scatter(f1[classification['dominated_invalid']], 
                   f2[classification['dominated_invalid']], 
                   alpha=0.4, s=30, color='#ff0000', marker='x',
                   label=f'Dominate non valide ({n_dominated_invalid})', zorder=1)
    
    if n_dominated_valid > 0:
        plt.scatter(f1[classification['dominated_valid']], 
                   f2[classification['dominated_valid']], 
                   alpha=0.4, s=30, color='#808080',
                   label=f'Dominate valide ({n_dominated_valid})', zorder=2)
    
    if n_pareto_invalid > 0:
        plt.scatter(f1[classification['pareto_invalid']], 
                   f2[classification['pareto_invalid']], 
                   alpha=0.6, s=50, color='#ffa500', marker='^',
                   label=f'Pareto non valide ({n_pareto_invalid})', zorder=3)
    
    if n_pareto_valid > 0:
        plt.scatter(f1[classification['pareto_valid']], 
                   f2[classification['pareto_valid']], 
                   alpha=0.6, s=50, color='#1f77b4',
                   label=f'Pareto valide ({n_pareto_valid})', zorder=4)
    
    # Se xs è fornito, evidenzia le soluzioni notevoli (SOLO tra quelle Pareto valide)
    if xs is not None and n_pareto_valid > 0:
        # Trova indici delle soluzioni Pareto valide
        pareto_valid_indices = np.where(classification['pareto_valid'])[0]
        
        # Trova le soluzioni notevoli SOLO tra quelle valide
        f1_valid = f1[classification['pareto_valid']]
        f2_valid = f2[classification['pareto_valid']]
        
        idx_best_comm_local = np.argmin(f1_valid)
        idx_best_cost_local = np.argmin(f2_valid)
        distances = np.sqrt(f1_valid**2 + f2_valid**2)
        idx_balanced_local = np.argmin(distances)
        
        # Converti in indici globali
        idx_best_comm = pareto_valid_indices[idx_best_comm_local]
        idx_best_cost = pareto_valid_indices[idx_best_cost_local]
        idx_balanced = pareto_valid_indices[idx_balanced_local]
        
        # Plot soluzioni notevoli
        plt.scatter(f1[idx_best_comm], f2[idx_best_comm], 
                   s=200, color='#ff7f0e', marker='D', 
                   edgecolors='black', linewidth=2,
                   label='A: Migliore Comunicazione', zorder=5)
        
        plt.scatter(f1[idx_best_cost], f2[idx_best_cost], 
                   s=200, color='#d62728', marker='D', 
                   edgecolors='black', linewidth=2,
                   label='B: Minor Costo', zorder=5)
        
        plt.scatter(f1[idx_balanced], f2[idx_balanced], 
                   s=200, color='#facf0a', marker='D', 
                   edgecolors='black', linewidth=2,
                   label='C: Bilanciata', zorder=5)
        
        # Calcola l'ipervolume
        hypervolume = calculate_hypervolume(ys, ref_point=[1.2, 1.4])
        
        # Aggiungi testo con ipervolume
        plt.text(0.98, 0.98, f'Hypervolume:\n{hypervolume:,.3f}', 
                transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.xlabel('Cost of communications between rovers and motherships — $J_1$', fontsize=12)
    plt.ylabel('Cost of building and operating the\ntwo satellite constellations — $J_2$', fontsize=12)
    plt.title('Fronte di Pareto - Classificazione Soluzioni', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Aggiungi punto di riferimento
    plt.axvline(x=1.2, color='r', linestyle='--', alpha=0.3, linewidth=1)
    plt.axhline(y=1.4, color='r', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    
    print(f"Grafico salvato in: {filename}")
    print(f"\nStatistiche soluzioni:")
    print(f"  - Pareto valide: {n_pareto_valid}")
    print(f"  - Pareto non valide: {n_pareto_invalid}")
    print(f"  - Dominate valide: {n_dominated_valid}")
    print(f"  - Dominate non valide: {n_dominated_invalid}")
    print(f"  - TOTALE: {len(ys)}")

# ============================================
# TROVA LE SOLUZIONI "MIGLIORI"
# ============================================

def find_best_solutions(xs, ys):
    """Identifica soluzioni notevoli nel fronte di Pareto (SOLO tra quelle valide)"""
    
    # Classifica le soluzioni
    classification = classify_solutions(ys)
    pareto_valid_mask = classification['pareto_valid']
    
    if not np.any(pareto_valid_mask):
        print("⚠️ Nessuna soluzione Pareto valida trovata!")
        return {}
    
    # Filtra solo le soluzioni Pareto valide
    pareto_valid_indices = np.where(pareto_valid_mask)[0]
    xs_valid = xs[pareto_valid_mask]
    ys_valid = ys[pareto_valid_mask]
    
    f1 = ys_valid[:, 0]
    f2 = ys_valid[:, 1]
    
    # 1. Soluzione con miglior comunicazione (minimo J1)
    idx_best_comm_local = np.argmin(f1)
    idx_best_comm = pareto_valid_indices[idx_best_comm_local]
    
    # 2. Soluzione con minor costo (minimo J2)
    idx_best_cost_local = np.argmin(f2)
    idx_best_cost = pareto_valid_indices[idx_best_cost_local]
    
    # 3. Soluzione "bilanciata" (minima distanza euclidea dall'origine)
    distances = np.sqrt(f1**2 + f2**2)
    idx_balanced_local = np.argmin(distances)
    idx_balanced = pareto_valid_indices[idx_balanced_local]
    
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
    """Esporta tutte le soluzioni in CSV con classificazione"""
    
    classification = classify_solutions(ys)
    
    data = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        params = decode_solution(x)
        
        # Determina categoria
        if classification['pareto_valid'][i]:
            categoria = 'Pareto_valida'
        elif classification['pareto_invalid'][i]:
            categoria = 'Pareto_non_valida'
        elif classification['dominated_valid'][i]:
            categoria = 'Dominata_valida'
        else:
            categoria = 'Dominata_non_valida'
        
        row = {
            'ID': i,
            'Categoria': categoria,
            'f1_comunicazione': y[0],
            'f2_costo': y[1],
            **params
        }
        
        # Aggiungi vincoli se presenti
        if y.shape[0] >= 4:
            row['vincolo_c1'] = y[2]
            row['vincolo_c2'] = y[3]
        
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
    filename = 'quantcomm_639739.npz'
    
    try:
        xs, ys = load_solutions(filename)
        print(f"✓ Caricate {len(xs)} soluzioni dal file {filename}")
        print(f"  - Dimensione xs: {xs.shape}")
        print(f"  - Dimensione ys: {ys.shape}\n")
        
        # VISUALIZZA IL FRONTE DI PARETO con classificazione
        plot_pareto_front(ys, xs)
        
        # TROVA SOLUZIONI NOTEVOLI
        print("\n" + "="*60)
        print("SOLUZIONI NOTEVOLI NEL FRONTE DI PARETO (solo valide)")
        print("="*60)
        
        best_sols = find_best_solutions(xs, ys)
        
        if best_sols:
            for name, (x, y) in best_sols.items():
                marker = 'A' if 'Comunicazione' in name else ('B' if 'Costo' in name else 'C')
                print(f"\n▶ [{marker}] {name}:")
                print(f"  Obiettivi: J1={y[0]:.6f}, J2={y[1]:.6f}")
                if y.shape[0] >= 4:
                    print(f"  Vincoli: c1={y[2]:.6f}, c2={y[3]:.6f}")
                params = decode_solution(x)
                
                print(f"\n  • Walker Constellation 1:")
                print(f"     - Satelliti totali: {params['W1_tot_satelliti']}")
                print(f"     - Configurazione: {params['W1_sat_per_piano']}×{params['W1_num_piani']} (phasing: {params['W1_phasing']})")
                print(f"     - Semiasse maggiore: {params['W1_semiasse_maggiore_km']:.2f} km")
                print(f"     - Eccentricità: {params['W1_eccentricita']:.6f}")
                print(f"     - Inclinazione: {params['W1_inclinazione_deg']:.2f}°")
                print(f"     - Argomento perigeo: {params['W1_arg_perigeo_deg']:.2f}°")
                print(f"     - Qualità η: {params['W1_qualita_eta']:.6f}")
                
                print(f"\n  • Walker Constellation 2:")
                print(f"     - Satelliti totali: {params['W2_tot_satelliti']}")
                print(f"     - Configurazione: {params['W2_sat_per_piano']}×{params['W2_num_piani']} (phasing: {params['W2_phasing']})")
                print(f"     - Semiasse maggiore: {params['W2_semiasse_maggiore_km']:.2f} km")
                print(f"     - Eccentricità: {params['W2_eccentricita']:.6f}")
                print(f"     - Inclinazione: {params['W2_inclinazione_deg']:.2f}°")
                print(f"     - Argomento perigeo: {params['W2_arg_perigeo_deg']:.2f}°")
                print(f"     - Qualità η: {params['W2_qualita_eta']:.6f}")
                
                print(f"\n  Rovers: {params['rover_indices']}")
                print(f"  Satelliti totali sistema: {params['satelliti_totali']}")
                print("-" * 60)
        
        # ESPORTA TUTTO IN CSV
        print("\n" + "="*60)
        df = export_to_csv(xs, ys)
        print(f"\nPrime 5 soluzioni Pareto valide:")
        df_pareto = df[df['Categoria'] == 'Pareto_valida']
        if len(df_pareto) > 0:
            print(df_pareto[['ID', 'f1_comunicazione', 'f2_costo', 'satelliti_totali']].head())
        else:
            print("Nessuna soluzione Pareto valida trovata!")
        
    except FileNotFoundError:
        print(f"❌ File {filename} non trovato!")
        print("   Assicurati che l'ottimizzazione abbia generato il file .npz")
