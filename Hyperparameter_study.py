"""Oefening 6: onderzoek de invloed van de hyperparameters.

Dit script herhaalt EXACT de pipeline van run_linear_regression.py (zelfde model,
criterion = MSELoss, optimizer = SGD, zelfde seed, zelfde volgorde van aanmaken,
zelfde evaluatie met het beste model "leader"), maar dan meerdere keren. Telkens
wordt EEN hyperparameter veranderd en blijft de rest op de basiswaarde staan.

Je kan dezelfde proeven ook met de hand doen: verander bovenaan in
run_linear_regression.py telkens een waarde en draai het bestand opnieuw.
Dit script doet dat gewoon in een keer en zet alles in een tabel en een figuur.

Uitvoeren (in dezelfde map als data.py, models.py, learners.py):
    python hyperparameter_study.py

Uitvoer: een tabel in de terminal en de figuur hyperparameter_study.png.
"""
import contextlib
import io
import math
import os

import matplotlib.pyplot as plt
import torch

from data import HousesDataset
from models import LinearRegressionModel
from learners import RegressionLearner

## BASISWAARDEN (identiek aan de PARAMETERS in run_linear_regression.py)

random_seed = 1
BASIS = dict(learning_rate=0.0001, batch_size=32, epochs=10, train_dataset_size=5000)
test_dataset_size = 20

## WAARDEN DIE ONDERZOCHT WORDEN (telkens 1 hyperparameter, de rest = BASIS)

EXPERIMENTEN = {
    "learning_rate":      [0.00001, 0.00003, 0.0001, 0.0003, 0.001],
    "batch_size":         [8, 32, 128, 512, 5000],
    "epochs":             [1, 5, 10, 30, 100],
    "train_dataset_size": [50, 200, 1000, 5000, 20000],
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LEADER_PAD = os.path.join("checkpoints", "LinearRegressionModel_leader")


def run(learning_rate, batch_size, epochs, train_dataset_size):
    """Een volledige run, net als run_linear_regression.py."""

    # INITIALISATION (zelfde als in run_linear_regression.py)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    train_dataset = HousesDataset(train_dataset_size)
    test_dataset = HousesDataset(test_dataset_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size)

    # MODEL DEFINITION
    model = LinearRegressionModel()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    learner = RegressionLearner(model, criterion, optimizer)

    # Verwijder een oud "leader"-bestand van een vorige run. Als de loss ontploft (nan),
    # wordt er geen nieuw bestand bewaard en zou je anders per ongeluk een oud model laden.
    if os.path.exists(LEADER_PAD):
        os.remove(LEADER_PAD)

    # TRAINING (de print per epoch wordt onderdrukt)
    with contextlib.redirect_stdout(io.StringIO()):
        train_losses = learner.train(train_dataloader, epochs, device)

    curve = [l if math.isfinite(l) else float("nan") for l in train_losses]

    # Ontploft = ergens in de training werd de loss nan of inf (of er is geen leader bewaard).
    if not all(math.isfinite(l) for l in train_losses) or not os.path.exists(LEADER_PAD):
        return dict(diverged=True, curve=curve)

    # TESTING (zelfde als in run_linear_regression.py: het beste model laden)
    learner.load("leader")
    return dict(
        diverged=False,
        curve=curve,
        train_rmse=learner.test(train_dataset, device),
        test_rmse=learner.test(test_dataset, device),
        a=model.linear_layer.weight.item(),
        b=model.linear_layer.bias.item(),
    )


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (naam, waarden) in zip(axes.flat, EXPERIMENTEN.items()):
    rest = {k: v for k, v in BASIS.items() if k != naam}
    print(f"\n=== {naam} veranderen (rest = {rest}) ===")
    print(f"{naam:>20} | train RMSE | test RMSE |   weight a |      bias b | epoch-1 loss -> laatste loss")

    for waarde in waarden:
        r = run(**{**BASIS, naam: waarde})
        c = r["curve"]
        if r["diverged"]:
            print(f"{waarde:>20} | {'nan':>10} | {'nan':>9} | {'nan':>10} | {'nan':>11} | de loss is ontploft (nan/inf)")
        else:
            print(f"{waarde:>20} | {r['train_rmse']:10.0f} | {r['test_rmse']:9.0f} | {r['a']:10.1f} | {r['b']:11.1f} | {c[0]:.3g} -> {c[-1]:.3g}")
        ax.plot(range(1, len(c) + 1), c, marker="o", markersize=3, label=f"{naam}={waarde}")

    ax.set_title(f"Trainingscurve bij verschillende {naam}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss (MSE)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)

print("\nReferentie: de ruis in de data is 50000 euro, dus de RMSE kan niet veel lager dan ~50000.")
print("Echte rechte: a = 5000, b = 100000.")

plt.tight_layout()
plt.savefig("hyperparameter_study.png", dpi=120)
plt.show()