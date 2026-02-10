# scenario.py
"""Multi-scenario engagement simulation"""

import numpy as np
from config import MISSILES, SCENARIOS, MAP_SIZE
from trajectory import MissileTrajectory


class Scenario:
    """Single scenario with aircraft firing at multiple targets"""
    
    def __init__(self, scenario_id, config, turbulence_grid):
        self.id = scenario_id
        self.config = config
        self.turbulence_grid = turbulence_grid
        
        self.aircraft_pos = np.array([30.0, MAP_SIZE[1] / 2])
        
        self.targets = []
        self.missiles = []
        self.trajectories = {}
        self.missile_counter = 0
        
        self.time = 0
        self.stats = {
            'missiles_launched': 0,
            'missiles_hit': 0,
            'avg_turbulence': 0,
            'avg_deviation': 0,
            'avg_cep': 0,
            'total_correction': 0,
            'impact_errors': []
        }
        
        self._create_targets()
    
    def _create_targets(self):
        """Create targets spread across the map"""
        num_targets = self.config['num_targets']
        
        for i in range(num_targets):
            x = 130 + np.random.uniform(0, 50)
            y = 25 + i * (MAP_SIZE[1] - 50) / max(num_targets - 1, 1)
            y += np.random.uniform(-8, 8)
            
            self.targets.append({
                'id': i,
                'position': np.array([x, y]),
                'status': 'active'
            })
    
    def launch_missiles(self):
        """Launch one missile at each target"""
        missile_type = self.config['missile_type']
        
        for target in self.targets:
            if target['status'] != 'active':
                continue
            
            missile_id = self.missile_counter
            self.missile_counter += 1
            
            trajectory = MissileTrajectory(
                missile_type, 
                self.turbulence_grid,
                use_correction=True,
                missile_id=missile_id
            )
            trajectory.launch(self.aircraft_pos, target['position'])
            
            self.missiles.append({
                'id': missile_id,
                'type': missile_type,
                'target_id': target['id'],
                'position': self.aircraft_pos.copy(),
                'altitude': 0.5,
                'status': 'active',
                'turbulence': 0,
                'deviation': 0,
                'mach': MISSILES[missile_type]['speed'],
                'correction': 0,
                'confidence': 0
            })
            
            self.trajectories[missile_id] = trajectory
            self.stats['missiles_launched'] += 1
    
    def update(self, dt):
        """Update all missiles"""
        self.time += dt
        
        turb_sum = 0
        dev_sum = 0
        corr_sum = 0
        active = 0
        
        for missile in self.missiles:
            if missile['status'] != 'active':
                continue
            
            mid = missile['id']
            if mid not in self.trajectories:
                continue
            
            traj = self.trajectories[mid]
            result = traj.step(dt)
            
            if result is None:
                continue
            
            if result['status'] == 'impact':
                missile['status'] = 'impact'
                missile['cep'] = result.get('cep', 0)
                missile['final_deviation'] = result.get('total_deviation', 0)
                missile['total_correction'] = result.get('total_correction', 0)
                missile['final_error'] = result.get('final_error', 0)
                
                self.stats['impact_errors'].append(result.get('final_error', 0))
                
                for t in self.targets:
                    if t['id'] == missile['target_id']:
                        t['status'] = 'destroyed'
                        self.stats['missiles_hit'] += 1
            else:
                missile['position'] = result['position'][:2]
                missile['altitude'] = result['altitude']
                missile['mach'] = result['mach']
                missile['turbulence'] = result['turbulence']
                missile['deviation'] = result['deviation']
                missile['correction'] = result['correction']
                missile['confidence'] = result.get('confidence', 0)
                
                turb_sum += result['turbulence']
                dev_sum += result['deviation']
                corr_sum += result['correction']
                active += 1
        
        if active > 0:
            self.stats['avg_turbulence'] = turb_sum / active
            self.stats['avg_deviation'] = dev_sum / active
            self.stats['total_correction'] = corr_sum
    
    def get_active_missiles(self):
        return [m for m in self.missiles if m['status'] == 'active']
    
    def is_complete(self):
        return len(self.get_active_missiles()) == 0 and self.stats['missiles_launched'] > 0
    
    def to_dict(self):
        """Serialize scenario for API"""
        def conv(p):
            return p.tolist() if hasattr(p, 'tolist') else list(p)
        
        missiles_data = []
        for m in self.missiles:
            mid = m['id']
            traj_points = None
            ideal_points = None
            
            if mid in self.trajectories:
                traj = self.trajectories[mid]
                traj_points = [conv(p) for p in traj.trajectory_history[-150:]]
                ideal_points = [conv(p) for p in traj.ideal_trajectory[-150:]]
            
            missiles_data.append({
                'id': m['id'],
                'type': m['type'],
                'position': conv(m['position']),
                'altitude': float(m['altitude']),
                'status': m['status'],
                'turbulence': float(m['turbulence']),
                'deviation': float(m['deviation']),
                'mach': float(m['mach']),
                'correction': float(m.get('correction', 0)),
                'confidence': float(m.get('confidence', 0)),
                'trajectory': traj_points,
                'ideal_trajectory': ideal_points,
                'cep': float(m.get('cep', 0)),
                'final_error': float(m.get('final_error', 0))
            })
        
        if self.stats['impact_errors']:
            sorted_errors = sorted(self.stats['impact_errors'])
            median_idx = len(sorted_errors) // 2
            self.stats['avg_cep'] = sorted_errors[median_idx]
        else:
            cep_values = [m.get('cep', 0) for m in self.missiles if m.get('cep', 0) > 0]
            self.stats['avg_cep'] = np.mean(cep_values) if cep_values else 0
        
        return {
            'id': self.id,
            'name': self.config['name'],
            'description': self.config['description'],
            'missile_type': self.config['missile_type'],
            'aircraft_pos': conv(self.aircraft_pos),
            'targets': [
                {'id': t['id'], 'position': conv(t['position']), 'status': t['status']}
                for t in self.targets
            ],
            'missiles': missiles_data,
            'time': float(self.time),
            'stats': {
                'missiles_launched': self.stats['missiles_launched'],
                'missiles_hit': self.stats['missiles_hit'],
                'avg_turbulence': float(self.stats['avg_turbulence']),
                'avg_deviation': float(self.stats['avg_deviation']),
                'avg_cep': float(self.stats['avg_cep']),
                'total_correction': float(self.stats['total_correction'])
            },
            'complete': self.is_complete()
        }


class MultiScenarioSimulation:
    """Manager for four parallel scenarios"""
    
    def __init__(self, turbulence_grid):
        self.turbulence_grid = turbulence_grid
        self.scenarios = {}
        self.running = False
        
        for scenario_id, config in SCENARIOS.items():
            self.scenarios[scenario_id] = Scenario(scenario_id, config, turbulence_grid)
    
    def launch_all(self):
        """Launch missiles in all scenarios simultaneously"""
        for scenario in self.scenarios.values():
            scenario.launch_missiles()
        self.running = True
    
    def update(self, dt):
        """Update all scenarios"""
        for scenario in self.scenarios.values():
            scenario.update(dt)
        
        if all(s.is_complete() for s in self.scenarios.values()):
            self.running = False
    
    def reset(self):
        """Reset all scenarios"""
        for scenario_id, config in SCENARIOS.items():
            self.scenarios[scenario_id] = Scenario(scenario_id, config, self.turbulence_grid)
        self.running = False
    
    def to_dict(self):
        """Serialize all scenarios"""
        return {
            'scenarios': {sid: s.to_dict() for sid, s in self.scenarios.items()},
            'running': self.running,
            'turbulence_profile': self.turbulence_grid.get_altitude_turbulence_profile(),
            'model_info': self.turbulence_grid.get_model_info()
        }