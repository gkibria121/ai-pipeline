
# ============================================================================
# FILE: utils/metrics.py
# ============================================================================

import os
import sys
import numpy as np


def compute_det_curve(target_scores, nontarget_scores):
    """Compute Detection Error Tradeoff (DET) curve"""
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((
        np.ones(target_scores.size),
        np.zeros(nontarget_scores.size)
    ))

    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((
        np.atleast_1d(0),
        tar_trial_sums / target_scores.size
    ))
    far = np.concatenate((
        np.atleast_1d(1),
        nontarget_trial_sums / nontarget_scores.size
    ))
    thresholds = np.concatenate((
        np.atleast_1d(all_scores[indices[0]] - 0.001),
        all_scores[indices]
    ))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """Compute Equal Error Rate (EER)"""
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """Compute ASV system error rates"""
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost=False):
    """
    Compute Tandem Detection Cost Function (t-DCF)
    
    Reference: Kinnunen et al., "t-DCF: a Detection Cost Function for 
    the Tandem Assessment of Spoofing Countermeasures and Automatic 
    Speaker Verification", Odyssey 2018
    """
    # Validate cost parameters
    if any(cost_model[k] < 0 for k in ['Cfa_asv', 'Cmiss_asv', 'Cfa_cm', 'Cmiss_cm']):
        print('WARNING: Cost values should be positive!')

    if any(cost_model[k] < 0 for k in ['Ptar', 'Pnon', 'Pspoof']) or \
       abs(sum(cost_model[k] for k in ['Ptar', 'Pnon', 'Pspoof']) - 1) > 1e-10:
        sys.exit('ERROR: Prior probabilities should be positive and sum to 1')

    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: Provide miss rate of spoof tests against ASV')

    # Validate scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Scores contain nan or inf')

    if np.unique(combined_scores).size < 3:
        sys.exit('ERROR: Provide soft scores, not binary decisions')

    # Compute CM error rates
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm
    )

    # Compute t-DCF
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - 
         cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    if C1 < 0 or C2 < 0:
        sys.exit('ERROR: Negative weights in t-DCF computation')

    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF / np.minimum(C1, C2)

    if print_cost:
        print(f't-DCF evaluation from [Nbona={bonafide_score_cm.size}, '
              f'Nspoof={spoof_score_cm.size}] trials\n')
        print('t-DCF MODEL')
        for key, desc in [
            ('Ptar', 'Prior probability of target user'),
            ('Pnon', 'Prior probability of nontarget user'),
            ('Pspoof', 'Prior probability of spoofing attack'),
            ('Cfa_asv', 'Cost of ASV falsely accepting nontarget'),
            ('Cmiss_asv', 'Cost of ASV falsely rejecting target'),
            ('Cfa_cm', 'Cost of CM falsely passing spoof'),
            ('Cmiss_cm', 'Cost of CM falsely blocking target')
        ]:
            print(f'   {key:12s} = {cost_model[key]:8.5f} ({desc})')

    return tDCF_norm, CM_thresholds


def calculate_tDCF_EER(cm_scores_file, asv_score_file, output_file, 
                       printout=True):
    """Calculate t-DCF and EER metrics"""
    # Cost model parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,
        'Pnon': (1 - Pspoof) * 0.01,
        'Cmiss': 1,
        'Cfa': 10,
        'Cmiss_asv': 1,
        'Cfa_asv': 10,
        'Cmiss_cm': 1,
        'Cfa_cm': 10,
    }

    # Load scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)

    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(float)

    # Extract scores by type
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # Compute metrics
    eer_asv, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    attack_types = [f'A{i:02d}' for i in range(7, 20)]
    
    if printout:
        spoof_cm_breakdown = {
            attack: cm_scores[cm_sources == attack]
            for attack in attack_types
        }
        eer_cm_breakdown = {
            attack: compute_eer(bona_cm, spoof_cm_breakdown[attack])[0]
            for attack in attack_types
        }

    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold
    )

    tDCF_curve, CM_thresholds = compute_tDCF(
        bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
        cost_model, print_cost=False
    )

    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    if printout:
        with open(output_file, "w") as f:
            f.write('\nCM SYSTEM\n')
            f.write(f'\tEER\t\t= {eer_cm * 100:8.9f} % '
                   '(Equal error rate)\n')
            f.write('\nTANDEM\n')
            f.write(f'\tmin-tDCF\t= {min_tDCF:8.9f}\n')
            f.write('\nBREAKDOWN CM SYSTEM\n')
            for attack in attack_types:
                eer_val = eer_cm_breakdown[attack] * 100
                f.write(f'\tEER {attack}\t= {eer_val:8.9f} %\n')
        
        os.system(f"cat {output_file}")

    return eer_cm * 100, min_tDCF

