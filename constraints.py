# constraints.py
"""Physics constraints for formation optimization"""

import numpy as np
from config import MIN_SEPARATION, MAX_SEPARATION, COMM_RANGE, MISSILES


def separation_penalty(positions):
    """Penalty for separation violations"""
    n = len(positions)
    if n < 2:
        return 0.0
    
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < MIN_SEPARATION:
                penalty += (MIN_SEPARATION - dist) ** 2 * 100
            if dist > MAX_SEPARATION:
                penalty += (dist - MAX_SEPARATION) ** 2 * 0.1
    return float(penalty)


def communication_penalty(positions):
    """Penalty for broken comms"""
    n = len(positions)
    if n < 2:
        return 0.0
    
    connected = [False] * n
    connected[0] = True
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not connected[i]:
                continue
            for j in range(n):
                if connected[j]:
                    continue
                if np.linalg.norm(positions[i] - positions[j]) <= COMM_RANGE:
                    connected[j] = True
                    changed = True
    
    return float(sum(1 for c in connected if not c) * 50.0)


def survivability_score(positions, threats):
    """Formation survivability against threats"""
    if not threats or len(positions) == 0:
        return 1.0
    
    total = 0.0
    for pos in positions:
        survival = 1.0
        for t in threats:
            t_pos = np.array(t['position'])
            dist = np.linalg.norm(pos - t_pos)
            if dist < t['range']:
                p_kill = t['pk'] * (1 - dist / t['range'])
                survival *= (1 - p_kill)
        total += survival
    
    return float(total / len(positions))


def weapon_coverage_score(positions, targets, weapons_list):
    """How well formation covers targets"""
    if not targets or len(positions) == 0:
        return 1.0
    
    total = 0.0
    for target in targets:
        t_pos = np.array(target['position'])
        best_pk = 0.0
        
        for i, pos in enumerate(positions):
            dist = np.linalg.norm(pos - t_pos)
            for wtype, count in weapons_list[i].items():
                if count <= 0:
                    continue
                m = MISSILES.get(wtype)
                if m and m['min_range'] <= dist <= m['range']:
                    range_factor = 1 - (dist - m['min_range']) / (m['range'] - m['min_range'])
                    pk = m['pk'] * (0.5 + 0.5 * range_factor)
                    best_pk = max(best_pk, pk)
        
        total += best_pk
    
    return float(total / len(targets))


def total_fitness(positions, weapons_list, targets, threats):
    """Total fitness score (higher = better)"""
    positions = np.array(positions)
    
    sep_pen = separation_penalty(positions)
    comm_pen = communication_penalty(positions)
    survival = survivability_score(positions, threats)
    coverage = weapon_coverage_score(positions, targets, weapons_list)
    
    return float(-sep_pen - comm_pen + survival * 100 + coverage * 80)