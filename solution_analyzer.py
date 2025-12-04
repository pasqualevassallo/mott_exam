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
