import numpy as np


def generate_sensor_data(n_samples: int = 500, seed: int = 42):
    """Sentetik makine sensörü verisi üretir.

    Özellikler: sicaklik (C), titresim (mm/s), bakimdan_gecen_saat
    Etiket: 1 = 7 gun icinde ariza riski yuksek, 0 = normal
    """
    rng = np.random.default_rng(seed)

    sicaklik = rng.normal(loc=60, scale=15, size=n_samples)
    titresim = rng.normal(loc=4, scale=2, size=n_samples)
    bakimdan_gecen_saat = rng.uniform(0, 2000, size=n_samples)

    risk_skoru = (
        (sicaklik - 60) / 15
        + (titresim - 4) / 2
        + (bakimdan_gecen_saat - 1000) / 1000
        + rng.normal(0, 0.5, size=n_samples)  # gürültü
    )
    etiket = (risk_skoru > 1.0).astype(int)

    X = np.column_stack([sicaklik, titresim, bakimdan_gecen_saat])
    y = etiket
    return X, y
