# config.py
"""Configuration for high-to-ground missile engagement with turbulence comparison"""

MAP_SIZE = (200, 200)  # km
ALTITUDE_LAYERS = [0, 0.5, 1, 2, 3, 5, 8, 12, 18, 25, 30]  # km

# ---- Physical constants ----
GRAVITY      = 9.81       # m/s²
EARTH_RADIUS = 6371.0     # km

MISSILES = {
    'SUBSONIC': {
        'name': 'Subsonic Cruise', 'speed': 0.85, 'cruise_altitude': 2,
        'mass': 450, 'drag_coeff': 0.025, 'ref_area': 0.35,
        'max_g_turn': 5, 'dive_angle': 45, 'guidance': 'TERCOM_GPS',
    },
    'SUPERSONIC': {
        'name': 'Supersonic', 'speed': 2.8, 'cruise_altitude': 10,
        'mass': 200, 'drag_coeff': 0.035, 'ref_area': 0.50,
        'max_g_turn': 8, 'dive_angle': 60, 'guidance': 'INS_ACTIVE',
    },
    'HIGH_SUPERSONIC': {
        'name': 'High Supersonic', 'speed': 4.5, 'cruise_altitude': 18,
        'mass': 280, 'drag_coeff': 0.030, 'ref_area': 0.40,
        'max_g_turn': 10, 'dive_angle': 65, 'guidance': 'INS_ACTIVE',
    },
    'HYPERSONIC': {
        'name': 'Hypersonic Glide', 'speed': 8.0, 'cruise_altitude': 25,
        'mass': 350, 'drag_coeff': 0.020, 'ref_area': 0.55,
        'max_g_turn': 3, 'dive_angle': 30, 'guidance': 'INS_STELLAR',
    },
}

SPEED_OF_SOUND_SEA   = 0.343    # km/s at sea level
AIR_DENSITY_SEA      = 1.225    # kg/m³
SCALE_HEIGHT         = 8.5      # km
KOLMOGOROV_CONSTANT  = 1.5
OUTER_SCALE          = 1.0      # km

# ---- Perturbation scales ----
TURB_SCALE  = 0.012
WIND_SCALE  = 0.0005

# ---- Guidance ----
GUIDANCE_BW_CRUISE   = 0.8      # Hz
GUIDANCE_BW_TERMINAL = 3.0      # Hz
DIVE_RANGE_FACTOR    = 2.5
TERMINAL_RANGE       = 5.0      # km

MIN_SEPARATION = 5.0
MAX_SEPARATION = 80.0
COMM_RANGE     = 60.0

SCENARIOS = {
    'LOW_ALT': {
        'name': 'Low Altitude (2 km)', 'missile_type': 'SUBSONIC',
        'launch_alt': 2, 'num_targets': 3,
        'description': 'High turbulence — boundary layer',
    },
    'MID_ALT': {
        'name': 'Mid Altitude (10 km)', 'missile_type': 'SUPERSONIC',
        'launch_alt': 10, 'num_targets': 3,
        'description': 'Moderate turbulence — troposphere',
    },
    'HIGH_ALT': {
        'name': 'High Altitude (18 km)', 'missile_type': 'HIGH_SUPERSONIC',
        'launch_alt': 18, 'num_targets': 3,
        'description': 'Low turbulence — tropopause',
    },
    'STRAT': {
        'name': 'Stratospheric (25 km)', 'missile_type': 'HYPERSONIC',
        'launch_alt': 25, 'num_targets': 3,
        'description': 'Minimal turbulence — stratosphere',
    },
}