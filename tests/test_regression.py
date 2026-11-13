"""Test regresyjny: stare ustawienia (wartości domyślne) + to samo ziarno = te same wyniki co PRZED zmianami etapu 2.

Pliki ref_legacy_*.npz zostały wygenerowane kodem sprzed zmian (N=400, 40 s, ziarno 42).
Symulacja jest chaotyczna, więc porównanie bit-w-bit ma sens tylko przy tych samych wersjach bibliotek;
jeśli po aktualizacji NumPy/SciPy/Numby test nie przechodzi, a liczba ewakuowanych jest zbliżona,
trzeba odtworzyć odniesienie starym kodem (git checkout) - nie oznacza to błędu modelu.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from _legacy_run import legacy_run  # noqa: E402


def _check(backend):
    ref = np.load(os.path.join(HERE, f"ref_legacy_{backend}.npz"))
    new = legacy_run(ROOT, backend)
    worst = max(float(np.abs(new[k] - ref[k]).max()) for k in ref.files)
    print(f"{backend}: maks. różnica względem kodu sprzed zmian = {worst:.3g}; "
          f"ewakuowani {int((new['state'] == 2).sum())} / {int((ref['state'] == 2).sum())}")
    assert worst < 1e-9, worst


def test_regression_numpy():
    _check("numpy")


def test_regression_numba():
    from evac.engine import HAVE_NUMBA
    if not HAVE_NUMBA:
        print("numba: pominięto (brak Numby)")
        return
    _check("numba")


if __name__ == "__main__":
    test_regression_numpy(); test_regression_numba(); print("OK")
