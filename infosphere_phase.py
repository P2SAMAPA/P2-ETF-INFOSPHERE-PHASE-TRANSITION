import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from scipy.signal import find_peaks

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def mutual_information_agent(returns, n_agents=5):
    """
    Compute mutual information between time-lagged "agents" in the infosphere.
    Agents are lagged versions of the return series.
    """
    if len(returns) < n_agents + 5:
        return 0.0
    # Create agent state matrix (lagged returns)
    agents = np.zeros((len(returns) - n_agents, n_agents))
    for i in range(n_agents):
        agents[:, i] = returns[i:-(n_agents - i)]
    # Compute pairwise mutual information (simplified: use correlation as proxy)
    corr_matrix = np.corrcoef(agents.T)
    # Mutual information is the average absolute correlation between agents
    # (In a true infosphere, this would be Shannon mutual information)
    mi = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
    return mi

def fisher_information_collective(returns, window=20):
    """
    Compute the Fisher information of the collective state.
    This measures how sensitive the distribution is to parameter changes.
    """
    if len(returns) < window + 5:
        return 0.0
    # Compute rolling volatility as the collective state parameter
    vol = np.zeros(len(returns) - window + 1)
    for i in range(len(vol)):
        vol[i] = np.std(returns[i:i+window])
    # Compute the derivative of the distribution (using histogram)
    hist, bin_edges = np.histogram(vol, bins=20, density=True)
    # Fisher information = sum (p_i * (d log p_i)^2)
    # Approximate derivative of log probability
    p = hist / (hist.sum() + 1e-8)
    dp = np.gradient(p)
    with np.errstate(divide='ignore', invalid='ignore'):
        fisher = p * (dp / (p + 1e-8))**2
    fisher = np.nan_to_num(fisher, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sum(fisher)

def phase_transition_order(returns, macro_factor, n_agents=5, fisher_window=20):
    """
    Compute the phase transition order parameter.
    High value = nearing criticality / regime shift.
    """
    if len(returns) < max(n_agents, fisher_window) + 5:
        return 0.0
    # 1. Mutual information between agents (infosphere coherence)
    mi = mutual_information_agent(returns, n_agents)
    # 2. Fisher information of collective state (sensitivity)
    fisher = fisher_information_collective(returns, fisher_window)
    # 3. Macro factor modulates the transition
    # 4. Combined order parameter
    # Normalise components
    mi_norm = min(mi, 1.0)
    fisher_norm = min(fisher / 10.0, 1.0)  # scale Fisher
    # Order parameter = geometric mean of coherence and sensitivity
    order = np.sqrt(mi_norm * fisher_norm)
    # Scale by macro factor
    order = order * (1 + macro_factor * 0.5)
    return order

def infosphere_score(returns, macro_df, n_agents=5, fisher_window=20):
    """
    Compute per-ETF infosphere phase transition score.
    Higher score = nearing criticality / regime shift.
    """
    if len(returns) < 25 or macro_df is None or len(macro_df) < 25:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 25:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Compute phase transition order parameter
    order = phase_transition_order(returns, macro_factor, n_agents, fisher_window)
    return float(order)
