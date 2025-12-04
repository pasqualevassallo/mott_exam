import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygmo as pg

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
# VISUALIZZA TUTTE LE SOLUZIONI E FRONTE DI PARETO
# ============================================

def plot_all_solutions_and_pareto(ys, xs=None, filename='pareto_analysis.png'):
    """Visualizza tutte le soluzioni e il fronte di Pareto separatamente"""
    
    # Crea figura con due subplot affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Estrai dati
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    constraints = ys[:, 2:] if ys.shape[1] > 2 else None
    
    # ============================================
    # GRAFICO 1: TUTTE LE SOLUZIONI
    # ============================================
    
    # Identifica diversi tipi di soluzioni
    if constraints is not None:
        # Soluzioni che violano i vincoli (c1 > 0 o c2 > 0)
        constraint_violations = np.any(constraints > 0, axis=1)
        valid_mask = ~constraint_violations
        
        # Soluzioni valide
        valid_f1 = f1[valid_mask]
        valid_f2 = f2[valid_mask]
        
        # Soluzioni che violano vincoli
        invalid_f1 = f1[constraint_violations]
        invalid_f2 = f2[constraint_violations]
        
        # Per le soluzioni valide, calcola il fronte di Pareto
        if len(valid_f1) > 0:
            valid_ys = np.column_stack([valid_f1, valid_f2])
            pareto_mask = pg.is_non_dominated(valid_ys.T)
            
            # Soluzioni valide ma non Pareto
            valid_nonpareto_f1 = valid_f1[~pareto_mask]
            valid_nonpareto_f2 = valid_f2[~pareto_mask]
            
            # Soluzioni Pareto
            pareto_f1 = valid_f1[pareto_mask]
            pareto_f2 = valid_f2[pareto_mask]
            
            # Plot soluzioni valide non Pareto
            if len(valid_nonpareto_f1) > 0:
                ax1.scatter(valid_nonpareto_f1, valid_nonpareto_f2, 
                           alpha=0.5, s=40, color='gray', 
                           label=f'Soluzioni valide dominate ({len(valid_nonpareto_f1)})')
            
            # Plot soluzioni Pareto
            if len(pareto_f1) > 0:
                ax1.scatter(pareto_f1, pareto_f2, 
                           alpha=0.8, s=80, color='blue', 
                           edgecolors='black', linewidth=1.5,
                           label=f'Fronte di Pareto ({len(pareto_f1)})', zorder=5)
        
        # Plot soluzioni che violano vincoli
        if len(invalid_f1) > 0:
            ax1.scatter(invalid_f1, invalid_f2, 
                       alpha=0.6, s=50, color='red', marker='x',
                       linewidths=1.5,
                       label=f'Violano vincoli ({len(invalid_f1)})', zorder=4)
    else:
        # Se non ci sono vincoli, plot tutte le soluzioni
        ax1.scatter(f1, f2, alpha=0.6, s=50, color='#1f77b4', 
                   label=f'Tutte le soluzioni ({len(f1)})')
    
    # Linee di riferimento
    ax1.axvline(x=1.2, color='r', linestyle='--', alpha=0.5, linewidth=1.5, 
                label='Riferimento f1=1.2')
    ax1.axhline(y=1.4, color='r', linestyle='--', alpha=0.5, linewidth=1.5,
                label='Riferimento f2=1.4')
    ax1.scatter(1.2, 1.4, s=100, color='red', marker='X', zorder=5, label='Punto riferimento')
    
    # Area obiettivo
    ax1.fill_between([0, 1.2], [0, 0], [1.4, 1.4], 
                     color='green', alpha=0.1, label='Area obiettivo')
    
    ax1.set_xlabel('Cost of communications between rovers and motherships — $J_1$', fontsize=12)
    ax1.set_ylabel('Cost of building and operating the\ntwo satellite constellations — $J_2$', fontsize=12)
    ax1.set_title('Tutte le Soluzioni - Classificazione', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # ============================================
    # GRAFICO 2: SOLO FRONTE DI PARETO
    # ============================================
    
    # Estrai solo le soluzioni Pareto se disponibili
    if constraints is not None and len(valid_f1) > 0 and 'pareto_f1' in locals() and len(pareto_f1) > 0:
        # Plot fronte di Pareto
        ax2.scatter(pareto_f1, pareto_f2, 
                   alpha=0.8, s=100, color='blue', 
                   edgecolors='black', linewidth=1.5,
                   label='Fronte di Pareto', zorder=5)
        
        # Ordina per f1 per una linea continua
        sorted_indices = np.argsort(pareto_f1)
        pareto_f1_sorted = pareto_f1[sorted_indices]
        pareto_f2_sorted = pareto_f2[sorted_indices]
        ax2.plot(pareto_f1_sorted, pareto_f2_sorted, 
                'b-', alpha=0.5, linewidth=2, zorder=2)
        
        # Evidenzia soluzioni notevoli se xs è fornito
        if xs is not None:
            # Trova gli indici delle soluzioni notevoli nel fronte di Pareto
            if len(pareto_f1) > 0:
                # Trova le soluzioni notevoli nel fronte di Pareto
                idx_best_comm = np.argmin(pareto_f1)
                idx_best_cost = np.argmin(pareto_f2)
                distances = np.sqrt(pareto_f1**2 + pareto_f2**2)
                idx_balanced = np.argmin(distances)
                
                # Plot soluzioni notevoli
                markers = [
                    ('D', '#ff7f0e', 'A: Migliore Comunicazione'),
                    ('s', '#d62728', 'B: Minor Costo'),
                    ('^', '#facf0a', 'C: Bilanciata')
                ]
                
                for idx, (marker, color, label) in zip([idx_best_comm, idx_best_cost, idx_balanced], markers):
                    ax2.scatter(pareto_f1[idx], pareto_f2[idx], 
                               s=200, color=color, marker=marker,
                               edgecolors='black', linewidth=2,
                               label=label, zorder=6)
    else:
        # Se non ci sono soluzioni Pareto, mostra messaggio
        ax2.text(0.5, 0.5, 'Nessuna soluzione Pareto-ottimale trovata',
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=14, color='red')
    
    # Linee di riferimento anche nel secondo grafico
    ax2.axvline(x=1.2, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axhline(y=1.4, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax2.scatter(1.2, 1.4, s=80, color='red', marker='X', zorder=5)
    
    ax2.set_xlabel('Cost of communications between rovers and motherships — $J_1$', fontsize=12)
    ax2.set_ylabel('Cost of building and operating the\ntwo satellite constellations — $J_2$', fontsize=12)
    ax2.set_title('Fronte di Pareto - Soluzioni Ottimali', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Grafico salvato in: {filename}")
    
    # Ritorna le informazioni per l'analisi successiva
    return {
        'all_f1': f1,
        'all_f2': f2,
        'pareto_f1': pareto_f1 if 'pareto_f1' in locals() else None,
        'pareto_f2': pareto_f2 if 'pareto_f2' in locals() else None,
        'constraint_violations': constraint_violations if 'constraint_violations' in locals() else None
    }

# ============================================
# TROVA LE SOLUZIONI "MIGLIORI" NEL FRONTE DI PARETO
# ============================================

def find_best_solutions(xs, ys):
    """Identifica soluzioni notevoli nel fronte di Pareto"""
    
    f1 = ys[:, 0]
    f2 = ys[:, 1]
    
    # Prima controlla se ci sono vincoli
    if ys.shape[1] > 2:
        constraints = ys[:, 2:]
        valid_mask = np.all(constraints <= 0, axis=1)
        valid_f1 = f1[valid_mask]
        valid_f2 = f2[valid_mask]
        
        if len(valid_f1) == 0:
            print("⚠️ Nessuna soluzione valida (tutte violano i vincoli)")
            return {}
        
        # Trova il fronte di Pareto tra le soluzioni valide
        valid_ys = np.column_stack([valid_f1, valid_f2])
        pareto_mask = pg.is_non_dominated(valid_ys.T)
        
        if not np.any(pareto_mask):
            print("⚠️ Nessuna soluzione Pareto-ottimale trovata")
            return {}
        
        pareto_f1 = valid_f1[pareto_mask]
        pareto_f2 = valid_f2[pareto_mask]
        
        # Usa gli indici corretti per xs
        valid_xs = xs[valid_mask]
        pareto_xs = valid_xs[pareto_mask]
        
        # Trova soluzioni notevoli nel fronte di Pareto
        idx_best_comm = np.argmin(pareto_f1)
        idx_best_cost = np.argmin(pareto_f2)
        distances = np.sqrt(pareto_f1**2 + pareto_f2**2)
        idx_balanced = np.argmin(distances)
        
        solutions = {
            'Migliore Comunicazione': (pareto_xs[idx_best_comm], 
                                      np.array([pareto_f1[idx_best_comm], 
                                               pareto_f2[idx_best_comm]])),
            'Minor Costo': (pareto_xs[idx_best_cost], 
                           np.array([pareto_f1[idx_best_cost], 
                                    pareto_f2[idx_best_cost]])),
            'Bilanciata': (pareto_xs[idx_balanced], 
                          np.array([pareto_f1[idx_balanced], 
                                   pareto_f2[idx_balanced]]))
        }
        
        return solutions
    else:
        # Se non ci sono vincoli, usa tutte le soluzioni
        idx_best_comm = np.argmin(f1)
        idx_best_cost = np.argmin(f2)
        distances = np.sqrt(f1**2 + f2**2)
        idx_balanced = np.argmin(distances)
        
        solutions = {
            'Migliore Comunicazione': (xs[idx_best_comm], ys[idx_best_comm, :2]),
            'Minor Costo': (xs[idx_best_cost], ys[idx_best_cost, :2]),
            'Bilanciata': (xs[idx_balanced], ys[idx_balanced, :2])
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
        }
        
        # Aggiungi vincoli se presenti
        if len(y) > 2:
            row['c1_vincolo_rover'] = y[2]
            row['c2_vincolo_satellite'] = y[3]
        
        row.update(params)
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
    filename = 'quantcomm_639739' 
    
    try:
        xs, ys = load_solutions(filename)
        print(f"✓ Caricate {len(xs)} soluzioni dal file {filename}")
        print(f"  - Dimensione xs: {xs.shape}")
        print(f"  - Dimensione ys: {ys.shape}")
        
        # Statistiche sui vincoli
        if ys.shape[1] > 2:
            constraints = ys[:, 2:]
            valid_solutions = np.sum(np.all(constraints <= 0, axis=1))
            print(f"  - Soluzioni valide (vincoli rispettati): {valid_solutions} ({valid_solutions/len(xs)*100:.1f}%)")
            print(f"  - Soluzioni che violano vincoli: {len(xs) - valid_solutions} ({(len(xs)-valid_solutions)/len(xs)*100:.1f}%)\n")
        
        # VISUALIZZA TUTTE LE SOLUZIONI E IL FRONTE DI PARETO
        plot_info = plot_all_solutions_and_pareto(ys, xs)
        
        # TROVA SOLUZIONI NOTEVOLI (nel fronte di Pareto)
        print("\n" + "="*60)
        print("SOLUZIONI NOTEVOLI NEL FRONTE DI PARETO")
        print("="*60)
        
        best_sols = find_best_solutions(xs, ys)
        
        if best_sols:
            for name, (x, y) in best_sols.items():
                marker = 'A' if 'Comunicazione' in name else ('B' if 'Costo' in name else 'C')
                print(f"\n▶ [{marker}] {name}:")
                print(f"  Obiettivi: f1={y[0]:.6f}, f2={y[1]:.6f}")
                params = decode_solution(x)
                
                print(f"\n • Walker Constellation 1:")
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
                print(f" Satelliti totali sistema: {params['satelliti_totali']}")
                print("-" * 60)
        else:
            print("\n⚠️ Non sono state trovate soluzioni Pareto-ottimali valide")
            print("   Le soluzioni mostrate nel grafico non rispettano i vincoli o non dominano il punto di riferimento")
        
        # ESPORTA TUTTO IN CSV
        print("\n" + "="*60)
        df = export_to_csv(xs, ys)
        
        # Informazioni aggiuntive
        if plot_info['pareto_f1'] is not None:
            print(f"\n📊 Dettaglio soluzioni Pareto:")
            print(f"   Numero di soluzioni Pareto-ottimali: {len(plot_info['pareto_f1'])}")
            print(f"   Range f1: [{plot_info['pareto_f1'].min():.4f}, {plot_info['pareto_f1'].max():.4f}]")
            print(f"   Range f2: [{plot_info['pareto_f2'].min():.4f}, {plot_info['pareto_f2'].max():.4f}]")
        
    except FileNotFoundError:
        print(f"❌ File {filename} non trovato!")
        print("   Assicurati che l'ottimizzazione abbia generato il file .npz")
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
