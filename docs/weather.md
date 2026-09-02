# Weather

`cascade.weather` gives an episode a mean wind with a boundary-layer profile, turbulence by
MIL-F-8785C class or by station record, and a gust generator that runs inside the step at the
aircraft's own altitude. `cascade.env` takes a `WeatherCondition` in `reset` and `step` (and
through `rollout_actions` and `rollout_policy`), so a batch of episodes can each fly in a
different draw of real weather.

## Condition

`weather_condition(wind_speed_m_s, wind_from_rad, turbulence_wind_20ft_m_s=None,
roughness_length_m=0.03)`: what a station reports. Speed and the direction the wind blows
from (clockwise from north) at 10 m; the 20 ft wind that sets MIL-F-8785C turbulence intensity
(by default the same speed; from records, the reported gust); the site's roughness length
(0.03 m open grass, 0.1 m to 0.5 m for scrub and suburbs, 0.001 m over water).

`weather_classes()` returns calm, light (7.7 m/s at 20 ft), moderate (15.4), and severe (23.2)
conditions with a matching mean wind from the north.

## Mean wind

`mean_wind_ned(condition, altitude_m)` applies the logarithmic profile
`V(h) = V10 · ln(h / z0) / ln(10 / z0)`, zero below the roughness length, and points the
vector where the wind blows to. An aircraft on final sees less wind than one at cruise height,
and a tailsitter hovering at 1.5 m sees about half the reported 10 m wind over grass.

## Gusts

`step_gust(condition, state, key, dt, airspeed_m_s, altitude_m, heading_rad)` advances the
Dryden filters of `cascade.gusts` one period with the intensities and length scales of the
aircraft's current altitude and airspeed, so turbulence changes as it climbs, descends, and
slows. The longitudinal gust acts along the heading, the lateral to its right, the vertical
down. Stationary intensity matches the class (`tests/test_weather.py`); a calm condition gives
exactly zero gust.

## Records

`WeatherRecords.from_csv(path)` reads hourly observations with columns `wind_speed_m_s`,
`wind_from_deg`, and optional `gust_m_s` (other columns are ignored, empty gusts fall back to
the wind speed), for example an export of a NOAA ISD or METAR history for a site.
`sample_weather(records, key)` draws one record as a condition: the mean wind from the
observation and the turbulence wind from its gust. `sample_weather_uniform(key)` is the
synthetic stand-in when no records are at hand. No station data ships with the package.

## In the environment

`reset(..., weather=condition)` shifts the initial ground velocity by the wind at the start
altitude so the aircraft begins at its trimmed airspeed rather than with a step, and stores
the wind and the gust filter state in `EnvState`. `step(..., weather=condition)` runs the
physics in the reference environment with that wind, then advances the gust for the next
period from the episode key. The observation's air data, the task's airspeed cost, and the
baseline controllers all see the current wind. Without a condition the reference's own wind
holds and nothing changes.

Under the cascade baseline on the aerobatic reference, a 6 m/s crosswind with moderate
turbulence (12 m/s turbulence wind) over a 4 s episode leaves the altitude within 5 m and the
mean reward above 0.5 once settled.
