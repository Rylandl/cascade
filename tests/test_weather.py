import jax
import jax.numpy as jnp
import numpy as np

from cascade.weather import (
    WeatherRecords,
    initial_gust_state,
    mean_wind_ned,
    sample_weather,
    sample_weather_uniform,
    step_gust,
    weather_classes,
    weather_condition,
)


def test_mean_wind_profile_and_direction_convention():
    north_wind = weather_condition(5.0, 0.0)
    at_reference = mean_wind_ned(north_wind, 10.0)
    # A wind from the north blows toward the south: negative north component, no east.
    assert float(at_reference[0]) < -4.99 and abs(float(at_reference[1])) < 1e-5
    assert float(mean_wind_ned(north_wind, 50.0)[0]) < float(at_reference[0])
    assert float(mean_wind_ned(north_wind, 2.0)[0]) > float(at_reference[0])
    assert float(jnp.linalg.norm(mean_wind_ned(north_wind, 0.0))) == 0.0
    west_wind = weather_condition(3.0, jnp.deg2rad(270.0))
    assert float(mean_wind_ned(west_wind, 10.0)[1]) > 2.99


def test_gust_generator_is_stationary_with_the_class_intensity():
    condition = weather_classes()["moderate"]
    dt = 0.02
    keys = jax.random.split(jax.random.PRNGKey(0), 6000)

    def scan_step(state, key):
        state, gust = step_gust(
            condition, state, key, dt, airspeed_m_s=15.0, altitude_m=50.0, heading_rad=0.0
        )
        return state, gust

    _, gusts = jax.lax.scan(scan_step, initial_gust_state(), keys)
    gusts = np.asarray(gusts[1000:])
    # sigma_w = 0.1 W20 = 1.54 m/s; sigma_u larger by the low-altitude ratio.
    assert 1.0 < gusts[:, 2].std() < 2.2
    assert gusts[:, 0].std() > gusts[:, 2].std()
    assert abs(gusts.mean(axis=0)).max() < 0.6
    calm = weather_classes()["calm"]
    _, still = step_gust(
        calm, initial_gust_state(), keys[0], dt, airspeed_m_s=15.0, altitude_m=50.0, heading_rad=0.0
    )
    assert float(jnp.max(jnp.abs(still))) == 0.0


def test_records_round_trip_csv_and_sample(tmp_path):
    path = tmp_path / "station.csv"
    path.write_text(
        "time,wind_speed_m_s,wind_from_deg,gust_m_s\n"
        "2025-01-01T00:00,3.0,180,\n"
        "2025-01-01T01:00,6.0,225,9.5\n"
        "2025-01-01T02:00,0.5,90,1.0\n"
    )
    records = WeatherRecords.from_csv(path)
    assert len(records) == 3
    assert float(records.gust_m_s[0]) == 3.0 and float(records.gust_m_s[1]) == 9.5
    condition = sample_weather(records, jax.random.PRNGKey(1))
    assert float(condition.wind_speed_m_s) in (3.0, 6.0, 0.5)
    assert float(condition.turbulence_wind_20ft_m_s) >= float(condition.wind_speed_m_s)
    synthetic = sample_weather_uniform(jax.random.PRNGKey(2))
    assert 0.0 <= float(synthetic.wind_speed_m_s) <= 12.0
