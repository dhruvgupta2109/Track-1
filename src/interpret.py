"""Physical interpretation of the c04 linear model (complete-data sub-model coefficients)."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, joblib
from features import BUILDINGS, TYPES, BUILDING_TYPE, clean_input, linear_design

def effects(art):
    """Marginal effects via finite differences of the design (robust to column layout)."""
    lin = art["linear"]; coef, icpt = lin["coefs"][()]
    rows = []
    for b in BUILDINGS:
        base = dict(building_id=b, building_type=BUILDING_TYPE[b], hour=12, day_of_week="Tuesday", month=6,
                    temperature=28.0, humidity=78.0, occupancy=50.0, previous_usage=50.0)
        f = lambda **kw: (linear_design(clean_input(pd.DataFrame([{**base, **kw}])), spec=lin["spec"]) @ coef + icpt)[0]
        rows.append(dict(building=b, type=BUILDING_TYPE[b],
                         per_100_occupants=f(occupancy=150.0) - f(),
                         per_degC_at_28=(f(temperature=28.5) - f(temperature=27.5)),
                         per_10_prev=f(previous_usage=60.0) - f(),
                         weekend_noon=f(day_of_week="Saturday") - f(),
                         empty_baseload_3am=f(hour=3, occupancy=0.0, previous_usage=0.0, temperature=26.0) ))
    return pd.DataFrame(rows).set_index("building")

def hour_profile(art, b, weekend=False):
    lin = art["linear"]; coef, icpt = lin["coefs"][()]
    df = pd.DataFrame([dict(building_id=b, building_type=BUILDING_TYPE[b], hour=h, day_of_week="Saturday" if weekend else "Tuesday",
                            month=6, temperature=28.0, humidity=78.0, occupancy=0.0, previous_usage=0.0) for h in range(24)])
    return linear_design(clean_input(df), spec=lin["spec"]) @ coef + icpt

if __name__ == "__main__":
    art = joblib.load("package/c04_linear_imputed/model.pkl")
    pd.set_option("display.width", 200); print(effects(art).round(2))
