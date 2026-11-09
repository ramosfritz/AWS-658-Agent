import copy
import math
import statistics
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error

def calculate_statistical_metrics(historical_data):
    """
    Compute comprehensive mathematical and statistical indicators on PCSO 6/58 data.
    """
    if not historical_data:
        return {}
        
    all_numbers = []
    sums = []
    odd_counts = 0
    even_counts = 0
    low_counts = 0   # 1-29
    high_counts = 0  # 30-58
    pair_counts = {}
    
    # Track gap / skip since last seen
    # historical_data is newest-first
    last_seen_gap = {}
    for num in range(1, 59):
        last_seen_gap[num] = None
        
    for draw_idx, draw in enumerate(historical_data):
        nums = sorted(draw['numbers'])
        all_numbers.extend(nums)
        draw_sum = sum(nums)
        sums.append(draw_sum)
        
        for n in nums:
            if last_seen_gap[n] is None:
                last_seen_gap[n] = draw_idx  # 0 means drawn in the most recent draw
            if n % 2 != 0:
                odd_counts += 1
            else:
                even_counts += 1
            if n <= 29:
                low_counts += 1
            else:
                high_counts += 1
                
        # Count pairs
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair = (nums[i], nums[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
    # Any numbers never seen in history get gap = len(historical_data)
    for num in range(1, 59):
        if last_seen_gap[num] is None:
            last_seen_gap[num] = len(historical_data)
            
    # Number frequency counts
    freq_series = pd.Series(all_numbers).value_counts()
    for num in range(1, 59):
        if num not in freq_series:
            freq_series[num] = 0
    freq_series = freq_series.sort_index()
    
    # Hot and Cold
    hot_numbers = freq_series.nlargest(6).index.tolist()
    cold_numbers = freq_series.nsmallest(6).index.tolist()
    
    # Overdue numbers (highest gap)
    gap_series = pd.Series(last_seen_gap)
    overdue_numbers = gap_series.nlargest(6).index.tolist()
    
    # Top pairs
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_pairs_formatted = [f"({p[0][0]}-{p[0][1]}: {p[1]}x)" for p in sorted_pairs]
    
    dates = [d['date'] for d in historical_data]
    min_date = min(dates).strftime('%b %d, %Y')
    max_date = max(dates).strftime('%b %d, %Y')
    
    total_balls = len(all_numbers) or 1
    odd_pct = (odd_counts / total_balls) * 100
    even_pct = (even_counts / total_balls) * 100
    low_pct = (low_counts / total_balls) * 100
    high_pct = (high_counts / total_balls) * 100
    
    return {
        'total_draws': len(historical_data),
        'min_date': min_date,
        'max_date': max_date,
        'frequency_series': freq_series,
        'gap_series': gap_series,
        'hot_numbers': hot_numbers,
        'cold_numbers': cold_numbers,
        'overdue_numbers': overdue_numbers,
        'mean_sum': float(np.mean(sums)) if sums else 177.0,
        'std_sum': float(np.std(sums)) if sums else 25.0,
        'median_sum': float(np.median(sums)) if sums else 177.0,
        'min_sum': int(np.min(sums)) if sums else 21,
        'max_sum': int(np.max(sums)) if sums else 333,
        'sums_list': sums,
        'odd_even_dist': f"{round(odd_pct)}% Odd : {round(even_pct)}% Even",
        'low_high_dist': f"{round(low_pct)}% Low (1-29) : {round(high_pct)}% High (30-58)",
        'top_pairs': top_pairs_formatted,
    }


class CDMPredictor:
    """
    Compound-Dirichlet-Multinomial (CDM) / Multivariate Polya Urn Model for 6/58 Lotto Prediction.
    Models the 58 lotto balls as a 58-dimensional categorical system with a conjugate Dirichlet prior
    Dir(alpha_1, ..., alpha_58) compounded with a Multinomial draw process.
    """
    def __init__(self, prior_strength=1.0, decay_factor=0.985, contagion_boost=0.15):
        self.num_balls = 58
        self.balls_per_draw = 6
        self.prior_strength = prior_strength
        self.decay_factor = decay_factor
        self.contagion_boost = contagion_boost
        self.is_trained = False

        # Learned parameters
        self.alpha_prior = np.ones(self.num_balls, dtype=float)
        self.alpha_posterior = np.ones(self.num_balls, dtype=float)
        self.posterior_probs = np.ones(self.num_balls, dtype=float) / self.num_balls
        self.posterior_std = np.zeros(self.num_balls, dtype=float)
        self.position_alphas = [np.ones(self.num_balls, dtype=float) for _ in range(self.balls_per_draw)]
        self.overdispersion = 0.0
        self.effective_sample_size = 0.0
        self.log_marginal_likelihood = 0.0
        self.empirical_alpha0 = 58.0

    def _estimate_empirical_bayes_alpha(self, count_matrix):
        """
        Estimate Dirichlet concentration parameter alpha_0 using Method of Moments / Empirical Bayes
        on historical draw indicator vectors (T x 58, each row summing to 6).
        """
        T, K = count_matrix.shape
        n = float(self.balls_per_draw)
        if T < 5:
            return np.ones(K, dtype=float) * self.prior_strength

        # Empirical mean and variance of proportions p_hat_{t, k} = x_{t, k} / n
        p_hat = count_matrix / n
        mean_p = np.clip(np.mean(p_hat, axis=0), 1e-4, 1.0 - 1e-4)
        mean_p = mean_p / np.sum(mean_p)
        sample_var = np.var(p_hat, axis=0, ddof=1)

        # Under CDM, E[s_k^2] = (p_k (1 - p_k) / n) * (1 + (n - 1) / (alpha_0 + 1))
        # Estimate total precision alpha_0 from average variance inflation across balls
        binom_var = mean_p * (1.0 - mean_p) / n
        ratio = np.sum(sample_var) / max(np.sum(binom_var), 1e-8)

        if ratio > 1.001:
            alpha_0 = (n - ratio) / (ratio - 1.0)
            alpha_0 = float(np.clip(alpha_0, 6.0, 580.0))
        else:
            alpha_0 = float(K * self.prior_strength)

        self.empirical_alpha0 = round(alpha_0, 3)
        return alpha_0 * mean_p

    def _compute_log_marginal_likelihood(self, count_matrix, alpha_vec):
        """
        Compute exact Dirichlet-Multinomial log marginal likelihood (log model evidence)
        across historical draws using log-gamma functions.
        """
        alpha_0 = float(np.sum(alpha_vec))
        n = float(self.balls_per_draw)
        log_lik = 0.0
        const_term = math.lgamma(alpha_0) - math.lgamma(alpha_0 + n)
        for row in count_matrix:
            row_term = const_term
            for k in np.where(row > 0)[0]:
                c_k = float(row[k])
                row_term += math.lgamma(alpha_vec[k] + c_k) - math.lgamma(alpha_vec[k])
            log_lik += row_term
        return round(log_lik, 2)

    def fit(self, historical_data, prior_strength=None, decay_factor=None, contagion_boost=None):
        """
        Fit the Compound-Dirichlet-Multinomial model to historical 6/58 draws (newest-first).
        Computes posterior Dirichlet concentrations, positional posteriors, overdispersion, and log-evidence.
        """
        if not historical_data:
            return False

        if prior_strength is not None:
            self.prior_strength = float(prior_strength)
        if decay_factor is not None:
            self.decay_factor = float(decay_factor)
        if contagion_boost is not None:
            self.contagion_boost = float(contagion_boost)

        T = len(historical_data)
        count_matrix = np.zeros((T, self.num_balls), dtype=float)
        pos_counts = [np.ones(self.num_balls, dtype=float) * 0.5 for _ in range(self.balls_per_draw)]
        last_seen_gap = np.full(self.num_balls, T, dtype=float)

        weighted_counts = np.zeros(self.num_balls, dtype=float)

        for t_idx, draw in enumerate(historical_data):
            nums = sorted(draw['numbers'])
            weight = (self.decay_factor ** t_idx)
            for pos_idx, ball in enumerate(nums):
                if 1 <= ball <= self.num_balls:
                    k = ball - 1
                    count_matrix[t_idx, k] = 1.0
                    weighted_counts[k] += weight
                    pos_counts[pos_idx][k] += weight
                    if last_seen_gap[k] == T:
                        last_seen_gap[k] = float(t_idx)

        # 1. Empirical Bayes prior concentration vector
        eb_prior = self._estimate_empirical_bayes_alpha(count_matrix)
        uniform_prior = np.ones(self.num_balls, dtype=float) * self.prior_strength
        self.alpha_prior = 0.5 * eb_prior + 0.5 * uniform_prior

        # 2. Polya contagion & hazard rebound modulation
        # Recent draws (t_idx < 5) trigger Polya urn contagion reinforcement;
        # Long gaps trigger mild Dirichlet hazard smoothing
        recent_burst = np.sum(count_matrix[:min(5, T)], axis=0)
        normalized_gap = last_seen_gap / max(float(T), 1.0)
        modulator = self.contagion_boost * recent_burst + 0.25 * normalized_gap

        # 3. Posterior Dirichlet concentration parameters: alpha_post = alpha_prior + weighted_counts + modulator
        self.alpha_posterior = self.alpha_prior + weighted_counts + modulator
        alpha_sum = float(np.sum(self.alpha_posterior))

        # 4. Posterior predictive mean E[p_k | D] and posterior standard deviation
        self.posterior_probs = self.alpha_posterior / alpha_sum
        self.posterior_std = np.sqrt(
            (self.posterior_probs * (1.0 - self.posterior_probs)) / (alpha_sum + 1.0)
        )

        # 5. Positional Dirichlet posteriors (for sorted Ball 1 .. Ball 6)
        self.position_alphas = [pos_c / np.sum(pos_c) * alpha_sum for pos_c in pos_counts]

        # 6. Diagnostic statistics
        self.effective_sample_size = round(alpha_sum, 2)
        self.overdispersion = round(1.0 / (alpha_sum + 1.0), 6)
        self.log_marginal_likelihood = self._compute_log_marginal_likelihood(count_matrix, self.alpha_prior)
        self.is_trained = True
        return True

    def predict(self, mode="map_bell_curve", temperature=1.0, rng_seed=None):
        """
        Generate a 6/58 prediction from the trained Compound-Dirichlet-Multinomial posterior.
        Modes:
          - 'map_bell_curve': Posterior MAP probabilities filtered for optimal 6/58 sum (145-205) & parity
          - 'map': Strictly the top 6 balls with highest posterior predictive probability E[p_k | D]
          - 'thompson': Dirichlet-Multinomial Thompson Sampling (draw p ~ Dir(alpha_post), then sample 6 balls)
          - 'polya_urn': Sequential Polya Urn draw with intra-draw contagion reinforcement
        """
        if not self.is_trained:
            return sorted(np.random.choice(range(1, 59), 6, replace=False).tolist())

        rng = np.random.default_rng(rng_seed)
        balls = np.arange(1, self.num_balls + 1)

        if mode == "map":
            top_indices = np.argsort(self.posterior_probs)[::-1][:6]
            return sorted(balls[top_indices].tolist())

        elif mode == "thompson":
            # Scale concentration by temperature (higher temp -> higher variance in Dirichlet sample)
            temp_scale = max(0.15, float(temperature))
            scaled_alpha = np.clip(self.alpha_posterior / temp_scale, 0.05, None)
            sampled_p = rng.dirichlet(scaled_alpha)
            sampled_p = sampled_p / np.sum(sampled_p)
            chosen = rng.choice(balls, size=6, replace=False, p=sampled_p)
            return sorted(chosen.tolist())

        elif mode == "polya_urn":
            # Sequential Urn sampling with decile spread and contagion weighting
            urn_weights = self.alpha_posterior.copy()
            selected = []
            for _ in range(6):
                probs = urn_weights / np.sum(urn_weights)
                pick = int(rng.choice(balls, p=probs))
                selected.append(pick)
                # Zero out picked ball so we sample without replacement, and mildly boost neighbors/decile balance
                urn_weights[pick - 1] = 0.0
            return sorted(selected)

        else:  # 'map_bell_curve' (Recommended Default)
            # Combine top posterior candidates with Dirichlet-Multinomial Monte Carlo rollouts
            # and pick the highest-likelihood combination satisfying 6/58 structural constraints
            top_pool_idx = np.argsort(self.posterior_probs)[::-1][:18]
            best_combo = None
            best_score = -1e9

            # Evaluate candidate combinations from the high-density Dirichlet region
            for _ in range(250):
                sampled_p = rng.dirichlet(self.alpha_posterior)
                cand = sorted(rng.choice(balls, size=6, replace=False, p=sampled_p).tolist())
                cand_sum = sum(cand)
                odds = sum(1 for n in cand if n % 2 != 0)
                lows = sum(1 for n in cand if n <= 29)

                # Log-predictive score under posterior mean
                log_prob = float(np.sum([math.log(max(self.posterior_probs[n - 1], 1e-8)) for n in cand]))

                # Structural penalties for sum outside [148, 202] or extreme parity
                sum_penalty = 0.0 if 148 <= cand_sum <= 202 else -abs(cand_sum - 177.0) * 0.03
                parity_bonus = 0.15 if odds in (2, 3, 4) else -0.4
                low_high_bonus = 0.15 if lows in (2, 3, 4) else -0.3

                total_score = log_prob + sum_penalty + parity_bonus + low_high_bonus
                if total_score > best_score:
                    best_score = total_score
                    best_combo = cand

            if best_combo is None:
                best_combo = sorted(balls[top_pool_idx[:6]].tolist())
            return best_combo

    def predict_positional_expectation(self):
        """Predict 6 unique numbers using positional Dirichlet posteriors (Ball 1 .. Ball 6)."""
        selected = []
        for pos in range(self.balls_per_draw):
            pos_probs = self.position_alphas[pos] / np.sum(self.position_alphas[pos])
            # Expected ball value or mode for this slot
            ranked = np.argsort(pos_probs)[::-1]
            for idx in ranked:
                candidate = int(idx + 1)
                if candidate not in selected:
                    selected.append(candidate)
                    break
        return sorted(selected)

    def predict_sequence(self, historical_data, steps=3):
        """
        Autoregressive Polya Urn multi-draw projection.
        After each predicted draw, reinforces the Dirichlet concentration vector (Polya contagion)
        while decaying older weights.
        """
        if not self.is_trained:
            self.fit(historical_data)

        alpha_curr = self.alpha_posterior.copy()
        balls = np.arange(1, self.num_balls + 1)
        sequences = []
        rng = np.random.default_rng()

        for step in range(steps):
            # Sample from evolving Dirichlet-Multinomial state
            best_cand = None
            best_score = -1e9
            for _ in range(120):
                p_draw = rng.dirichlet(alpha_curr)
                cand = sorted(rng.choice(balls, size=6, replace=False, p=p_draw).tolist())
                cand_sum = sum(cand)
                odds = sum(1 for n in cand if n % 2 != 0)
                score = float(np.sum([math.log(max(p_draw[n - 1], 1e-8)) for n in cand]))
                if 150 <= cand_sum <= 200 and odds in (2, 3, 4):
                    score += 0.35
                if score > best_score:
                    best_score = score
                    best_cand = cand

            odds = sum(1 for n in best_cand if n % 2 != 0)
            sequences.append({
                "draw_step": step + 1,
                "numbers": best_cand,
                "sum": sum(best_cand),
                "odd_even": f"{odds}:{6 - odds}",
                "strategy_focus": f"CDM Polya Urn Step {step + 1} (α₀={round(float(np.sum(alpha_curr)), 1)})"
            })

            # Polya Urn transition update: decay + reinforce drawn numbers
            alpha_curr = alpha_curr * self.decay_factor
            for n in best_cand:
                alpha_curr[n - 1] += (1.0 + self.contagion_boost)

        return sequences

    def get_diagnostics(self):
        """Return structured summary of learned CDM parameters and top balls."""
        balls = np.arange(1, self.num_balls + 1)
        top_idx = np.argsort(self.posterior_probs)[::-1][:10]
        top_balls = [int(balls[i]) for i in top_idx]
        top_probs = [round(float(self.posterior_probs[i]) * 100, 3) for i in top_idx]

        return {
            "model_name": "Compound-Dirichlet-Multinomial (CDM)",
            "prior_strength": self.prior_strength,
            "decay_factor": self.decay_factor,
            "contagion_boost": self.contagion_boost,
            "empirical_alpha0": self.empirical_alpha0,
            "effective_sample_size": self.effective_sample_size,
            "overdispersion": self.overdispersion,
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "top_balls": top_balls,
            "top_probs_pct": top_probs,
            "posterior_probs_series": pd.Series(self.posterior_probs, index=balls),
            "alpha_posterior_series": pd.Series(self.alpha_posterior, index=balls),
            "posterior_std_series": pd.Series(self.posterior_std, index=balls),
        }


class ClassicalMLPredictor:
    """
    Classical Machine Learning & Probabilistic models suite for 6/58 draw forecasting.
    Includes Compound-Dirichlet-Multinomial (CDM), Random Forest, Gradient Boosting,
    Support Vector Classifier, and MLP Neural Network.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVC': SVC(kernel='rbf', C=1.0),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.cdm_model = CDMPredictor()
        self.trained_models = {}
        self.feature_columns = []
        self.is_trained = False
        self.performance_summary = []
        self.rf_importances = []
        
    def prepare_features(self, data):
        """Prepare lag and rolling statistical features from newest-first data."""
        features_list = []
        for i in range(len(data) - 3):
            target = data[i]
            prev1 = data[i + 1]
            prev2 = data[i + 2]
            prev3 = data[i + 3]
            
            row = {
                'd1': target['numbers'][0],
                'd2': target['numbers'][1],
                'd3': target['numbers'][2],
                'd4': target['numbers'][3],
                'd5': target['numbers'][4],
                'd6': target['numbers'][5],
                # Lag 1
                'prev_d1': prev1['numbers'][0],
                'prev_d2': prev1['numbers'][1],
                'prev_d3': prev1['numbers'][2],
                'prev_d4': prev1['numbers'][3],
                'prev_d5': prev1['numbers'][4],
                'prev_d6': prev1['numbers'][5],
                'prev_sum': sum(prev1['numbers']),
                # Lag 2
                'prev2_d1': prev2['numbers'][0],
                'prev2_d2': prev2['numbers'][1],
                'prev2_d3': prev2['numbers'][2],
                'prev2_d4': prev2['numbers'][3],
                'prev2_d5': prev2['numbers'][4],
                'prev2_d6': prev2['numbers'][5],
                'prev2_sum': sum(prev2['numbers']),
                # Lag 3
                'prev3_d1': prev3['numbers'][0],
                'prev3_d2': prev3['numbers'][1],
                'prev3_d3': prev3['numbers'][2],
                'prev3_d4': prev3['numbers'][3],
                'prev3_d5': prev3['numbers'][4],
                'prev3_d6': prev3['numbers'][5],
                'prev3_sum': sum(prev3['numbers']),
                # Rolling stats
                'roll_mean_d1': np.mean([prev1['numbers'][0], prev2['numbers'][0], prev3['numbers'][0]]),
                'roll_mean_d2': np.mean([prev1['numbers'][1], prev2['numbers'][1], prev3['numbers'][1]]),
                'roll_mean_d3': np.mean([prev1['numbers'][2], prev2['numbers'][2], prev3['numbers'][2]]),
                'roll_mean_d4': np.mean([prev1['numbers'][3], prev2['numbers'][3], prev3['numbers'][3]]),
                'roll_mean_d5': np.mean([prev1['numbers'][4], prev2['numbers'][4], prev3['numbers'][4]]),
                'roll_mean_d6': np.mean([prev1['numbers'][5], prev2['numbers'][5], prev3['numbers'][5]]),
                'roll_std_d1': float(np.std([prev1['numbers'][0], prev2['numbers'][0], prev3['numbers'][0]])),
                'roll_std_d2': float(np.std([prev1['numbers'][1], prev2['numbers'][1], prev3['numbers'][1]])),
                'roll_std_d3': float(np.std([prev1['numbers'][2], prev2['numbers'][2], prev3['numbers'][2]])),
                'roll_std_d4': float(np.std([prev1['numbers'][3], prev2['numbers'][3], prev3['numbers'][3]])),
                'roll_std_d5': float(np.std([prev1['numbers'][4], prev2['numbers'][4], prev3['numbers'][4]])),
                'roll_std_d6': float(np.std([prev1['numbers'][5], prev2['numbers'][5], prev3['numbers'][5]])),
                # Calendar features
                'day_of_week': pd.Timestamp(target['date']).weekday(),
                'month': pd.Timestamp(target['date']).month,
                'day': pd.Timestamp(target['date']).day,
                'is_weekend': 1 if pd.Timestamp(target['date']).weekday() >= 5 else 0,
            }
            features_list.append(row)
            
        return pd.DataFrame(features_list)
        
    def train_models(self, historical_data):
        """Train scikit-learn models and Compound-Dirichlet-Multinomial (CDM) model on historical data."""
        if not historical_data or len(historical_data) < 20:
            return False

        # Train Compound-Dirichlet-Multinomial model
        self.cdm_model.fit(historical_data)
            
        df = self.prepare_features(historical_data)
        self.feature_columns = [col for col in df.columns if col not in ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']]
        
        X = df[self.feature_columns]
        y = df[['d1', 'd2', 'd3', 'd4', 'd5', 'd6']]
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.trained_models = {}
        for model_name, base_model in self.models.items():
            digit_models = []
            for digit in range(6):
                y_digit = y.iloc[:, digit]
                m = copy.deepcopy(base_model)
                m.fit(X_scaled, y_digit)
                digit_models.append(m)
            self.trained_models[model_name] = digit_models
            
        # Calculate MAE metrics for Classical ML + CDM
        self.performance_summary = []
        for model_name, digit_models in self.trained_models.items():
            maes = [round(mean_absolute_error(y.iloc[:, j], digit_models[j].predict(X_scaled)), 3) for j in range(6)]
            self.performance_summary.append({
                'Model': model_name,
                'Ball 1': maes[0], 'Ball 2': maes[1], 'Ball 3': maes[2],
                'Ball 4': maes[3], 'Ball 5': maes[4], 'Ball 6': maes[5],
                'Avg MAE': round(float(np.mean(maes)), 3)
            })

        # Evaluate CDM positional expectation + rolling Dirichlet shrinkage MAE
        balls_arr = np.arange(1, 59)
        cdm_maes = []
        for j in range(6):
            pos_probs = self.cdm_model.position_alphas[j] / np.sum(self.cdm_model.position_alphas[j])
            pos_expected = float(np.sum(balls_arr * pos_probs))
            roll_col = f'roll_mean_d{j+1}'
            # Bayesian Dirichlet shrinkage combining positional Dirichlet expectation & local trajectory
            cdm_pred_series = 0.55 * pos_expected + 0.45 * df[roll_col]
            cdm_maes.append(round(mean_absolute_error(y.iloc[:, j], cdm_pred_series), 3))

        self.performance_summary.append({
            'Model': 'CDM (Dirichlet-Multinomial)',
            'Ball 1': cdm_maes[0], 'Ball 2': cdm_maes[1], 'Ball 3': cdm_maes[2],
            'Ball 4': cdm_maes[3], 'Ball 5': cdm_maes[4], 'Ball 6': cdm_maes[5],
            'Avg MAE': round(float(np.mean(cdm_maes)), 3)
        })
            
        # Extract Random Forest feature importances
        if 'Random Forest' in self.trained_models:
            rf0 = self.trained_models['Random Forest'][0]
            if hasattr(rf0, 'feature_importances_'):
                self.rf_importances = rf0.feature_importances_.tolist()
                
        self.is_trained = True
        return True

    def _extract_next_features(self, historical_data, target_date):
        """Build feature vector for the upcoming draw."""
        latest = historical_data[0]
        prev2 = historical_data[1]
        prev3 = historical_data[2]
        
        features = {
            'prev_d1': latest['numbers'][0],
            'prev_d2': latest['numbers'][1],
            'prev_d3': latest['numbers'][2],
            'prev_d4': latest['numbers'][3],
            'prev_d5': latest['numbers'][4],
            'prev_d6': latest['numbers'][5],
            'prev_sum': sum(latest['numbers']),
            'prev2_d1': prev2['numbers'][0],
            'prev2_d2': prev2['numbers'][1],
            'prev2_d3': prev2['numbers'][2],
            'prev2_d4': prev2['numbers'][3],
            'prev2_d5': prev2['numbers'][4],
            'prev2_d6': prev2['numbers'][5],
            'prev2_sum': sum(prev2['numbers']),
            'prev3_d1': prev3['numbers'][0],
            'prev3_d2': prev3['numbers'][1],
            'prev3_d3': prev3['numbers'][2],
            'prev3_d4': prev3['numbers'][3],
            'prev3_d5': prev3['numbers'][4],
            'prev3_d6': prev3['numbers'][5],
            'prev3_sum': sum(prev3['numbers']),
            'roll_mean_d1': np.mean([latest['numbers'][0], prev2['numbers'][0], prev3['numbers'][0]]),
            'roll_mean_d2': np.mean([latest['numbers'][1], prev2['numbers'][1], prev3['numbers'][1]]),
            'roll_mean_d3': np.mean([latest['numbers'][2], prev2['numbers'][2], prev3['numbers'][2]]),
            'roll_mean_d4': np.mean([latest['numbers'][3], prev2['numbers'][3], prev3['numbers'][3]]),
            'roll_mean_d5': np.mean([latest['numbers'][4], prev2['numbers'][4], prev3['numbers'][4]]),
            'roll_mean_d6': np.mean([latest['numbers'][5], prev2['numbers'][5], prev3['numbers'][5]]),
            'roll_std_d1': float(np.std([latest['numbers'][0], prev2['numbers'][0], prev3['numbers'][0]])),
            'roll_std_d2': float(np.std([latest['numbers'][1], prev2['numbers'][1], prev3['numbers'][1]])),
            'roll_std_d3': float(np.std([latest['numbers'][2], prev2['numbers'][2], prev3['numbers'][2]])),
            'roll_std_d4': float(np.std([latest['numbers'][3], prev2['numbers'][3], prev3['numbers'][3]])),
            'roll_std_d5': float(np.std([latest['numbers'][4], prev2['numbers'][4], prev3['numbers'][4]])),
            'roll_std_d6': float(np.std([latest['numbers'][5], prev2['numbers'][5], prev3['numbers'][5]])),
            'day_of_week': pd.Timestamp(target_date).weekday(),
            'month': target_date.month,
            'day': target_date.day,
            'is_weekend': 1 if pd.Timestamp(target_date).weekday() >= 5 else 0,
        }
        df_feat = pd.DataFrame([features])
        return self.scaler.transform(df_feat[self.feature_columns])

    def predict_next(self, historical_data, variance=0.0):
        """Predict next draw using classical models, CDM, and hybrid ensemble."""
        if not self.is_trained or len(historical_data) < 3:
            return None
            
        latest_date = historical_data[0]['date']
        target_date = latest_date + timedelta(days=1)
        while target_date.weekday() not in [6, 1, 4]:  # Sunday(6), Tuesday(1), Friday(4)
            target_date += timedelta(days=1)
            
        scaled_feat = self._extract_next_features(historical_data, target_date)
        if variance > 0.0:
            scaled_feat = scaled_feat + np.random.normal(0, variance, scaled_feat.shape)
            
        predictions = {}
        for model_name, digit_models in self.trained_models.items():
            pred_digits = []
            for digit in range(6):
                pred = int(digit_models[digit].predict(scaled_feat)[0])
                while pred in pred_digits or pred < 1 or pred > 58:
                    pred = int(np.random.randint(1, 59))
                pred_digits.append(pred)
            predictions[model_name] = sorted(pred_digits)

        # Compound-Dirichlet-Multinomial (CDM) Predictions
        if not self.cdm_model.is_trained:
            self.cdm_model.fit(historical_data)
        predictions['CDM (Dirichlet-Multinomial)'] = self.cdm_model.predict(mode="map_bell_curve")
        predictions['CDM Thompson Sample'] = self.cdm_model.predict(mode="thompson", temperature=1.0 + variance)
        predictions['CDM Posterior MAP'] = self.cdm_model.predict(mode="map")
            
        # Classical + CDM Consensus Ensemble
        ensemble = []
        ensemble_models = [
            'Random Forest',
            'Gradient Boosting',
            'SVC',
            'Neural Network',
            'CDM (Dirichlet-Multinomial)'
        ]
        for digit in range(6):
            candidates = [predictions[m][digit] for m in ensemble_models if m in predictions]
            try:
                pick = int(statistics.mode(candidates))
            except statistics.StatisticsError:
                # Tie-break using highest CDM posterior probability among candidates
                cdm_probs = self.cdm_model.posterior_probs
                pick = max(candidates, key=lambda b: cdm_probs[b - 1] if 1 <= b <= 58 else 0.0)
            while pick in ensemble or pick < 1 or pick > 58:
                # Pick highest-probability unused ball from CDM posterior
                ranked_balls = np.argsort(self.cdm_model.posterior_probs)[::-1] + 1
                for rb in ranked_balls:
                    if int(rb) not in ensemble:
                        pick = int(rb)
                        break
            ensemble.append(pick)
            
        predictions['Classic Ensemble'] = sorted(ensemble)
        return predictions

    def predict_sequence(self, historical_data, sequence_length=3):
        """Autoregressive multi-draw sequence prediction."""
        if not self.is_trained or len(historical_data) < 3:
            return []
            
        current_data = copy.deepcopy(historical_data[:10])
        predicted_sequences = []
        
        for step in range(sequence_length):
            preds = self.predict_next(current_data, variance=0.1 * step)
            if not preds:
                break
            pick = preds.get('Classic Ensemble', preds.get('Random Forest'))
            predicted_sequences.append(pick)
            
            # Autoregressive step: prepend predicted draw
            next_date = current_data[0]['date'] + timedelta(days=1)
            while next_date.weekday() not in [6, 1, 4]:
                next_date += timedelta(days=1)
                
            new_draw = {
                'numbers': pick,
                'date': next_date,
                'jackpot': '₱0.00',
                'winners': 0,
                'is_sample': True
            }
            current_data.insert(0, new_draw)
            
        return predicted_sequences
