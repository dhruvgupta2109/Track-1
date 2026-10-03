"""Named model specs (stored inside the artifact, so the notebook never needs this file)."""
import copy
from features import DEFAULT_SPEC
V2 = copy.deepcopy(DEFAULT_SPEC)  # c02
V3 = copy.deepcopy(DEFAULT_SPEC)  # c03
V3.update(hour_fourier=3, weekend_hours="shared",
          missing_spec=dict(hour_fourier=6, weekend_hours="type"))
V4 = copy.deepcopy(V3)            # + shrunk per-hour deviations per type
V4["hour_dev_scale"] = 0.25
V5 = copy.deepcopy(V4)            # + residential day/night steps, stronger hour-deviation shrinkage (from submission 007)
V5["hour_dev_scale"] = 0.1
V5["res_steps"] = True
