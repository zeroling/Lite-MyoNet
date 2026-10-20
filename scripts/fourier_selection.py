"""Choose the main Fourier degree from completed paired validation runs."""
import json
import math
import statistics
from pathlib import Path

CANDIDATES = ('temporal_fourier_d1', 'temporal_fourier_d2', 'temporal_fourier_d3')


def choose_degree(state, seeds):
    by_id = {item['id']: item for item in state['items']}
    reports = []
    for degree, item_id in enumerate(CANDIDATES, 1):
        item = by_id[item_id]
        if item.get('status') != 'completed':
            raise RuntimeError(f'候选配置尚未完成：{item_id}')
        values = []
        runs = []
        for seed in seeds:
            record = item['seeds'][str(seed)]
            if record.get('status') != 'completed':
                raise RuntimeError(f'候选种子尚未完成：{item_id}/{seed}')
            run = Path(record['run_dir'])
            result = json.loads((run / 'test_results.json').read_text(encoding='utf-8'))
            if int(result['seed']) != int(seed) or result['selection_metric'] != 'accuracy':
                raise ValueError(f'候选选模信息不匹配：{run}')
            # This field is the best validation score, independent of test metrics.
            score = float(result['selection_value'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'候选验证指标无效：{run}')
            values.append(score)
            runs.append({'seed': int(seed), 'run_dir': str(run), 'best_validation_accuracy': score,
                         'best_epoch': int(result['checkpoint_epoch'])})
        reports.append({'degree': degree, 'candidate_id': item_id,
                        'validation_mean': statistics.mean(values),
                        'validation_sample_std': statistics.stdev(values), 'runs': runs})
    winner = max(reports, key=lambda row: (row['validation_mean'], -row['degree']))
    return {'selected_degree': winner['degree'], 'selected_candidate': winner['candidate_id'],
            'metric': 'best_validation_accuracy_mean', 'seeds': list(seeds),
            'tie_break': 'lower_degree', 'candidates': reports}


def apply_selection(plan, state, selection):
    plan['selected_fourier_degree'] = int(selection['selected_degree'])
    for item in plan['items']:
        if item.get('kind') == 'reference':
            item['reference_to'] = selection['selected_candidate']
            for saved in state['items']:
                if saved['id'] == item['id']:
                    saved['reference_to'] = selection['selected_candidate']
                    saved['source_item'] = dict(item)
    state['fourier_selection'] = selection
