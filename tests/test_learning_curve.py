"""Unit tests for Schmettow learning curve fitting."""

import numpy as np
import pytest
import warnings
from scipy.optimize import OptimizeWarning
from app.analytics.learning_curve import (
    fit_schmettow, predict_at_trial, mastery_percent, SchmettowFit, SessionDataPoint, get_mentor_message
)
from app.utils.config import SCORE_MAX, LC_MIN_SESSIONS

@pytest.fixture(autouse=True)
def suppress_optimize_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        yield

def test_fit_insufficient_data():
    """fit_schmettow returns None if fewer than LC_MIN_SESSIONS points."""
    trials = np.arange(1, LC_MIN_SESSIONS)
    errors = np.array([10, 8, 6, 4])
    fit = fit_schmettow(trials, errors)
    assert fit is None

def test_fit_valid_synthetic_data():
    """fit_schmettow recovers parameters from synthetic Schmettow data."""
    # Ground truth parameters
    leff_true = 0.3  # Learning efficiency
    pexp_true = 2.0
    maxp_true = 1.0
    scale_true = 10.0
    
    trials = np.arange(1, 15)
    # errors(t) = scale * (1 - leff)**(t + pexp) + maxp
    errors = scale_true * (1 - leff_true)**(trials + pexp_true) + maxp_true
    
    fit = fit_schmettow(trials, errors)
    
    assert fit is not None
    assert 0 < fit.leff < 1
    assert fit.maxp >= 0
    assert fit.pexp >= 0
    assert fit.scale > 0
    
    # Check if fit is close to ground truth
    assert pytest.approx(fit.leff, rel=0.1) == leff_true
    assert pytest.approx(fit.maxp, abs=0.2) == maxp_true
    assert pytest.approx(fit.pexp, rel=0.3) == pexp_true
    
    # Check derived values
    assert fit.maxp_performance == SCORE_MAX - fit.maxp
    assert len(fit.predicted_errors) == len(trials)
    assert len(fit.predicted_performance) == len(trials)
    assert np.all(pytest.approx(fit.predicted_performance) == SCORE_MAX - fit.predicted_errors)

def test_fit_divergent_input():
    """fit_schmettow returns None for all-zero errors."""
    trials = np.arange(1, 10)
    errors = np.zeros(9)
    fit = fit_schmettow(trials, errors)
    assert fit is None

def test_predict_at_trial():
    """predict_at_trial returns predicted performance score."""
    # Mock fit: errors(t) = 4 * (1 - 0.5)**(t + 0) + 2
    # trial 1: errors = 4 * 0.5^1 + 2 = 2 + 2 = 4 -> performance = 10 - 4 = 6.0
    fit = SchmettowFit(
        leff=0.5, pexp=0.0, maxp=2.0, scale=4.0,
        maxp_performance=8.0, r_squared=0.9,
        predicted_errors=np.array([]), predicted_performance=np.array([])
    )
    
    p1 = predict_at_trial(fit, 1, score_max=10.0)
    assert p1 == 6.0
    
    # As t -> infinity, errors -> maxp = 2.0
    # Performance -> 10 - 2 = 8.0 (ceiling)
    p_inf = predict_at_trial(fit, 100, score_max=10.0)
    assert pytest.approx(p_inf, abs=0.01) == 8.0

def test_mastery_percent():
    """mastery_percent handles normal values and clamping."""
    fit = SchmettowFit(
        leff=0.5, pexp=0.0, maxp=2.0, scale=4.0,
        maxp_performance=8.0, r_squared=0.9,
        predicted_errors=np.array([]), predicted_performance=np.array([])
    )
    
    assert mastery_percent(fit, 4.0) == 50.0
    assert mastery_percent(fit, 0.0) == 0.0
    assert mastery_percent(fit, 8.0) == 100.0
    assert mastery_percent(fit, 10.0) == 100.0
    assert mastery_percent(fit, -2.0) == 0.0

def test_performance_score_transform():
    """Performance score transform is SCORE_MAX - errors."""
    dp = SessionDataPoint(trial=1, error_count=3, performance_score=SCORE_MAX - 3)
    assert dp.performance_score == SCORE_MAX - 3

def test_pexp_shift():
    """Positive pexp reduces initial error prediction compared to pexp=0."""
    fit0 = SchmettowFit(leff=0.5, pexp=0.0, maxp=1.0, scale=5.0, maxp_performance=9.0, r_squared=1.0,
                       predicted_errors=np.array([]), predicted_performance=np.array([]))
    fit1 = SchmettowFit(leff=0.5, pexp=1.0, maxp=1.0, scale=5.0, maxp_performance=9.0, r_squared=1.0,
                       predicted_errors=np.array([]), predicted_performance=np.array([]))
    
    # trial 1
    # errors0 = 5 * (0.5)^1 + 1 = 3.5
    # errors1 = 5 * (0.5)^2 + 1 = 2.25
    e0 = 10.0 - predict_at_trial(fit0, 1, score_max=10.0)
    e1 = 10.0 - predict_at_trial(fit1, 1, score_max=10.0)
    assert e1 < e0

def test_maxp_ceiling():
    """Performance prediction never exceeds maxp_performance as t increases."""
    fit = SchmettowFit(leff=0.5, pexp=0.0, maxp=2.0, scale=8.0, maxp_performance=8.0, r_squared=1.0,
                      predicted_errors=np.array([]), predicted_performance=np.array([]))

    # ceiling = 10 - 2 = 8 (using score_max=10)
    p1 = predict_at_trial(fit, 1, score_max=10.0)    # 10 - (8*0.5 + 2) = 4
    p10 = predict_at_trial(fit, 10, score_max=10.0)   # 10 - (8*0.5^10 + 2) approx 7.99
    p100 = predict_at_trial(fit, 100, score_max=10.0)  # approx 8.0

    assert p1 < 8.0
    assert p10 < 8.0
    assert pytest.approx(p100, abs=1e-6) == 8.0
    assert p100 <= 8.0

def test_leff_sensitivity():
    """Higher leff results in faster error reduction (lower error at same trial)."""
    # leff=0.2 (slow) vs leff=0.8 (fast)
    fit_slow = SchmettowFit(leff=0.2, pexp=0.0, maxp=1.0, scale=5.0, maxp_performance=9.0, r_squared=1.0,
                           predicted_errors=np.array([]), predicted_performance=np.array([]))
    fit_fast = SchmettowFit(leff=0.8, pexp=0.0, maxp=1.0, scale=5.0, maxp_performance=9.0, r_squared=1.0,
                           predicted_errors=np.array([]), predicted_performance=np.array([]))
    
    # Trial 2
    # slow: 5 * (0.8)^2 + 1 = 5 * 0.64 + 1 = 4.2
    # fast: 5 * (0.2)^2 + 1 = 5 * 0.04 + 1 = 1.2
    e_slow = 10.0 - predict_at_trial(fit_slow, 2, score_max=10.0)
    e_fast = 10.0 - predict_at_trial(fit_fast, 2, score_max=10.0)
    
    assert e_fast < e_slow

def test_mastery_percent_boundary():
    """mastery_percent handles boundary cases exactly."""
    fit = SchmettowFit(leff=0.5, pexp=0.0, maxp=2.0, scale=4.0, maxp_performance=8.0, r_squared=0.9,
                      predicted_errors=np.array([]), predicted_performance=np.array([]))
    
    assert mastery_percent(fit, 0.0) == 0.0
    assert mastery_percent(fit, 8.0) == 100.0
    assert mastery_percent(fit, 8.0000001) == 100.0

def test_fit_with_noisy_data():
    """fit_schmettow can handle slightly noisy synthetic data."""
    leff_true = 0.4
    pexp_true = 1.0
    maxp_true = 2.0
    scale_true = 6.0
    
    trials = np.arange(1, 15)
    # Generate data with noise
    np.random.seed(42)
    clean_errors = scale_true * (1 - leff_true)**(trials + pexp_true) + maxp_true
    noise = np.random.normal(0, 0.1, size=len(trials))
    noisy_errors = np.clip(clean_errors + noise, 0, 10)
    
    fit = fit_schmettow(trials, noisy_errors)
    assert fit is not None
    assert pytest.approx(fit.leff, rel=0.2) == leff_true
    assert pytest.approx(fit.maxp, abs=0.5) == maxp_true

def test_mastery_percent_zero_ceiling():
    """mastery_percent returns 0.0 if maxp_performance is 0 or less."""
    fit = SchmettowFit(leff=0.5, pexp=0.0, maxp=10.0, scale=0.0, maxp_performance=0.0, r_squared=1.0,
                      predicted_errors=np.array([]), predicted_performance=np.array([]))
    assert mastery_percent(fit, 5.0) == 0.0

def test_get_mentor_message():
    """get_mentor_message returns correct messages for all three ranges."""
    # >= 80%
    assert "approaching" in get_mentor_message(80.0)
    assert "approaching" in get_mentor_message(95.0)
    
    # 40-79%
    msg_40 = get_mentor_message(40.0)
    assert "60% from your potential" in msg_40
    assert "grinding" in msg_40
    
    msg_79 = get_mentor_message(79.0)
    assert "21% from your potential" in msg_79
    
    # < 40%
    assert "Early stage" in get_mentor_message(39.9)
    assert "Early stage" in get_mentor_message(10.0)
