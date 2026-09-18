# Offline flight inspection

`cascade.viz.trajectory_report` turns a saved trajectory into a standalone HTML report.
It needs the core package only: no MuJoCo, plotting package, web service, or network connection.

```python
from cascade.viz import trajectory_report

trajectory_report("flight.npz", "flight.html")
trajectory_report(
    "baseline.npz",
    "comparison.html",
    compare="candidate.npz",
    labels=("Baseline", "Candidate"),
    title="Controller comparison",
)
```

The report contains altitude and ground-speed plots, an east/north flight path, a selectable
signal plot, a shared timeline cursor with playback, fault/event markers, and full file metadata.
Signals include body rates, actuator deflections, propeller speeds, separation and applied
native controls. The experiment runner's `.diagnostics.npz` sidecar adds commanded/applied
normalized actions, reward, airspeed, tracking error, and sensor age/validity.
Runner airspeed and tracking-error diagnostics describe simulator truth, not the policy's
noisy or delayed measurements. Sensor age and validity describe what the policy received.

Ages are unknown until a valid sample arrives; these readings appear as gaps, with validity
available as a separate signal. A selected commanded-action signal also overlays its applied
action with a dashed line. Existing action latency and actuator dynamics remain distinct:
normalized applied actions are the inputs reaching the actuators, not their achieved positions.

Time axes use stored timestamps. Comparison never silently shifts recordings or interpolates
one recording onto the other. Controls and actions belong to the interval ending at each
stored timestamp. Scheduled events outside a recording are labeled and cannot move the cursor.
The cursor shows the nearest displayed sample and reports no
value outside that flight's recorded range. For long flights, `max_points=2500` bounds display
size and preserves endpoints; the original trajectory remains the numerical source.

Reports accept one nonempty, unbatched trajectory per input. Select a batch member before
saving its inspection file. Ground speed is labeled separately from airspeed: wind is not
inferred when a standalone trajectory has no environment diagnostics.
