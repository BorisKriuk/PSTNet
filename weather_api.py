# weather_api.py
"""Real-time weather from NASA POWER API — bugs fixed"""

import numpy as np
from datetime import datetime, timedelta
import json
import urllib.request

FALLBACK_WEATHER = {
    'wind_speed_10m': 6.0, 'wind_speed_50m': 10.0,
    'temperature_2m': 288.0, 'pressure_surface': 101.3,
    'humidity': 55.0, 'precipitation': 0.0, 'cloud_cover': 40.0,
}


class WeatherService:
    def __init__(self, lat=35.0, lon=-120.0):
        self.lat = lat
        self.lon = lon
        self.cache = {}
        self.last_fetch = None
        self.fetch_interval = 3600

    # ---- fetch ----------------------------------------------------------
    def fetch_nasa_power(self):
        end = datetime.now()
        start = end - timedelta(days=7)
        qs = (
            f"parameters=WS10M,WS50M,T2M,PS,RH2M,PRECTOTCORR,CLOUD_AMT"
            f"&community=RE&longitude={self.lon}&latitude={self.lat}"
            f"&start={start:%Y%m%d}&end={end:%Y%m%d}&format=JSON"
        )
        url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + qs
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            props = data.get('properties', {}).get('parameter', {})

            def latest(pd, fb):
                for d in sorted(pd.keys(), reverse=True):
                    if pd[d] != -999:
                        return float(pd[d])
                return fb

            w = {
                'wind_speed_10m':    latest(props.get('WS10M', {}), FALLBACK_WEATHER['wind_speed_10m']),
                'wind_speed_50m':    latest(props.get('WS50M', {}), FALLBACK_WEATHER['wind_speed_50m']),
                'temperature_2m':    latest(props.get('T2M', {}), 15.0) + 273.15,
                'pressure_surface':  latest(props.get('PS', {}), FALLBACK_WEATHER['pressure_surface']),
                'humidity':          latest(props.get('RH2M', {}), FALLBACK_WEATHER['humidity']),
                'precipitation':     latest(props.get('PRECTOTCORR', {}), FALLBACK_WEATHER['precipitation']),
                'cloud_cover':       latest(props.get('CLOUD_AMT', {}), FALLBACK_WEATHER['cloud_cover']),
            }
            self.cache = w
            self.last_fetch = datetime.now()
            print(f"NASA weather: wind={w['wind_speed_10m']:.1f} m/s  T={w['temperature_2m']-273.15:.1f} °C")
            return w
        except Exception as e:
            print(f"NASA API error: {e} — using fallback")
            return FALLBACK_WEATHER.copy()

    # ---- cached access --------------------------------------------------
    def get_weather(self):
        now = datetime.now()
        if self.last_fetch is None or (now - self.last_fetch).total_seconds() > self.fetch_interval:
            return self.fetch_nasa_power()
        return self.cache.copy() if self.cache else FALLBACK_WEATHER.copy()

    # ---- vertical profile (ISA + measured surface) ----------------------
    def get_vertical_profile(self, altitudes):
        w = self.get_weather()
        profile = []
        for alt in altitudes:
            # Temperature — ISA lapse
            if alt < 11:
                temp = w['temperature_2m'] - 6.5 * alt
            elif alt < 20:
                temp = w['temperature_2m'] - 6.5 * 11
            else:
                temp = w['temperature_2m'] - 6.5 * 11 + 1.0 * (alt - 20)
            temp = max(temp, 180.0)

            # Pressure — barometric
            pres = w['pressure_surface'] * np.exp(-alt / 8.5)
            pres = max(pres, 0.001)

            # Density — ideal gas
            density = (pres * 1000) / (287.05 * temp)
            density = max(density, 1e-5)

            # Wind — realistic profile
            sw = w['wind_speed_10m']
            if alt < 1:
                wind = sw * (1 + 0.3 * alt)
            elif alt < 10:
                wind = sw * 1.3 + (alt - 1) * 1.2
            elif alt < 15:
                wind = sw * 1.3 + 9 * 1.2 + 6 * max(0, 1 - abs(alt - 12) / 3)
            else:
                wind = max(sw * 1.3 + 9 * 1.2 - (alt - 15) * 0.6, 1.0)
            wind = float(np.clip(wind, 0.5, 80))

            # Richardson number
            if alt < 11:    dT = -6.5
            elif alt < 20:  dT = 0.0
            else:           dT = 1.0
            ws = max(wind / max(alt, 0.1), 0.1)
            ri = float(np.clip((9.81 / temp) * (dT + 9.8) / (ws**2 + 0.01), -5, 10))

            profile.append(dict(altitude=float(alt), temperature=float(temp),
                                pressure=float(pres), density=float(density),
                                wind_speed=wind, richardson=ri))
        return profile