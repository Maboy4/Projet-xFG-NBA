import matplotlib.pyplot as plt


def plot_loss_comparison(loss_lr: float, loss_xgb: float) -> None:
    """
    Graphique barre simple : comparaison Log Loss des deux modeles.
    """
    modeles = ['Hasard', 'Reg. Logistique', 'XGBoost']
    valeurs = [0.6931, loss_lr, loss_xgb]
    couleurs = ['#d9534f', '#f0ad4e', '#5cb85c']

    plt.figure(figsize=(7, 5))
    bars = plt.bar(modeles, valeurs, color=couleurs, width=0.5)

    for bar, val in zip(bars, valeurs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.002,
                 f'{val:.4f}',
                 ha='center', fontsize=11)

    plt.ylim(0.63, 0.71)
    plt.ylabel('Log Loss (plus bas = mieux)')
    plt.title('Comparaison des modeles — Log Loss')
    plt.tight_layout()
    plt.show()