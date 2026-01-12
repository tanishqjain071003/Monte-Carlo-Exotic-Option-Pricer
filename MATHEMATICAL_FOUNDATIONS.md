# Mathematical Foundations of Options Pricing & Greeks

## Table of Contents
1. [Geometric Brownian Motion (GBM)](#geometric-brownian-motion-gbm)
2. [Monte Carlo Simulation](#monte-carlo-simulation)
3. [Black-Scholes-Merton (BSM) Model](#black-scholes-merton-bsm-model)
4. [Option Greeks](#option-greeks)
5. [Finite Difference Method for Greeks](#finite-difference-method-for-greeks)
6. [Common Random Numbers (CRN)](#common-random-numbers-crn)
7. [Stress Testing & Scenario Analysis](#stress-testing--scenario-analysis)
8. [Exotic Options Payoffs](#exotic-options-payoffs)

---

## 1. Geometric Brownian Motion (GBM)

### Mathematical Definition

The GBM is a stochastic process used to model stock prices under the risk-neutral measure:

\[
dS_t = r S_t dt + \sigma S_t dW_t
\]

Where:
- \(S_t\) = Stock price at time \(t\)
- \(r\) = Risk-free interest rate (annualized)
- \(\sigma\) = Volatility (annualized)
- \(dW_t\) = Wiener process (Brownian motion increment)

### Solution via Itô's Lemma

Applying Itô's Lemma to \(\ln(S_t)\), we get:

\[
d(\ln S_t) = \left(r - \frac{\sigma^2}{2}\right) dt + \sigma dW_t
\]

Integrating from \(0\) to \(T\):

\[
\ln S_T - \ln S_0 = \left(r - \frac{\sigma^2}{2}\right) T + \sigma W_T
\]

Exponentiating:

\[
S_T = S_0 \exp\left[\left(r - \frac{\sigma^2}{2}\right) T + \sigma W_T\right]
\]

### Discrete-Time Implementation

For Monte Carlo simulation, we discretize time into \(N\) steps:

\[
S_{t+\Delta t} = S_t \exp\left[\left(r - \frac{\sigma^2}{2}\right) \Delta t + \sigma \sqrt{\Delta t} Z\right]
\]

Where:
- \(\Delta t = T/N\) = time step size
- \(Z \sim \mathcal{N}(0,1)\) = standard normal random variable
- \(N\) = number of time steps

### Path Generation Formula

For a path with \(N\) steps:

\[
S_{t_i} = S_0 \exp\left[\sum_{j=1}^{i} \left(r - \frac{\sigma^2}{2}\right) \Delta t + \sigma \sqrt{\Delta t} Z_j\right]
\]

Where \(Z_j\) are independent standard normal random variables.

**Key Properties:**
- **Drift term**: \((r - \sigma^2/2)\Delta t\) ensures risk-neutral pricing
- **Diffusion term**: \(\sigma \sqrt{\Delta t} Z\) captures volatility
- **Log-normal distribution**: Stock prices are log-normally distributed
- **Martingale property**: Under risk-neutral measure, \(E[S_T] = S_0 e^{rT}\)

---

## 2. Monte Carlo Simulation

### Option Pricing Formula

The risk-neutral pricing formula:

\[
V_0 = e^{-rT} \mathbb{E}^Q[Payoff(S_T)]
\]

For Monte Carlo, we approximate the expectation:

\[
V_0 \approx e^{-rT} \frac{1}{M} \sum_{i=1}^{M} Payoff(S_T^{(i)})
\]

Where:
- \(M\) = number of simulation paths
- \(S_T^{(i)}\) = terminal stock price in path \(i\)

### Standard Error

The standard error of the Monte Carlo estimate:

\[
SE = \frac{\sigma_{payoff}}{\sqrt{M}}
\]

Where \(\sigma_{payoff}\) is the standard deviation of payoffs across all paths.

### Convergence Rate

Monte Carlo converges at rate \(O(1/\sqrt{M})\):
- To halve the error, need 4× more paths
- Central Limit Theorem: estimate is asymptotically normal

### Variance Reduction: Antithetic Variates

For each path \(Z\), also simulate \(-Z\):

\[
\text{Payoff}_1 = f(S_T(Z)), \quad \text{Payoff}_2 = f(S_T(-Z))
\]

\[
\text{Estimate} = \frac{1}{2}(\text{Payoff}_1 + \text{Payoff}_2)
\]

This reduces variance because payoffs are negatively correlated.

---

## 3. Black-Scholes-Merton (BSM) Model

### BSM Formula for European Options

**Call Option:**

\[
C = S_0 N(d_1) - K e^{-rT} N(d_2)
\]

**Put Option:**

\[
P = K e^{-rT} N(-d_2) - S_0 N(-d_1)
\]

Where:

\[
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma \sqrt{T}}
\]

\[
d_2 = d_1 - \sigma \sqrt{T} = \frac{\ln(S_0/K) + (r - \sigma^2/2)T}{\sigma \sqrt{T}}
\]

And \(N(\cdot)\) is the cumulative distribution function of the standard normal distribution.

### Derivation Intuition

1. **Risk-neutral valuation**: Price = discounted expected payoff
2. **Log-normal distribution**: Under GBM, \(\ln(S_T/S_0) \sim \mathcal{N}((r-\sigma^2/2)T, \sigma^2 T)\)
3. **Integration**: \(N(d_1)\) and \(N(d_2)\) come from integrating the log-normal PDF

### Key Assumptions

- Constant volatility \(\sigma\)
- Constant risk-free rate \(r\)
- No dividends
- Continuous trading
- No transaction costs
- Log-normal stock price distribution

---

## 4. Option Greeks

Greeks measure sensitivity of option price to various parameters.

### Delta (Δ)

**Definition:** Sensitivity to underlying price change

\[
\Delta = \frac{\partial V}{\partial S}
\]

**BSM Formula:**
- Call: \(\Delta_{call} = N(d_1)\)
- Put: \(\Delta_{put} = N(d_1) - 1 = -N(-d_1)\)

**Interpretation:**
- Delta ≈ probability option expires in-the-money (for calls)
- Delta hedging: Long \(\Delta\) shares to hedge short option
- Range: Call delta ∈ [0, 1], Put delta ∈ [-1, 0]

### Gamma (Γ)

**Definition:** Rate of change of Delta (second derivative)

\[
\Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{\partial \Delta}{\partial S}
\]

**BSM Formula:**

\[
\Gamma = \frac{n(d_1)}{S_0 \sigma \sqrt{T}}
\]

Where \(n(d_1) = \frac{1}{\sqrt{2\pi}} e^{-d_1^2/2}\) is the standard normal PDF.

**Interpretation:**
- Measures convexity of option price
- Always positive (for both calls and puts)
- Highest for at-the-money options
- Important for delta hedging accuracy

### Vega (ν)

**Definition:** Sensitivity to volatility change

\[
\nu = \frac{\partial V}{\partial \sigma}
\]

**BSM Formula:**

\[
\nu = S_0 n(d_1) \sqrt{T}
\]

**Interpretation:**
- Usually expressed per 1% volatility change: \(\nu/100\)
- Always positive (both calls and puts)
- Higher for longer-dated, at-the-money options

### Rho (ρ)

**Definition:** Sensitivity to interest rate change

\[
\rho = \frac{\partial V}{\partial r}
\]

**BSM Formula:**
- Call: \(\rho_{call} = K T e^{-rT} N(d_2)\)
- Put: \(\rho_{put} = -K T e^{-rT} N(-d_2)\)

**Interpretation:**
- Usually expressed per 1% rate change: \(\rho/100\)
- Calls: positive (higher rates → higher call prices)
- Puts: negative (higher rates → lower put prices)

### Theta (Θ)

**Definition:** Time decay (sensitivity to time passage)

\[
\Theta = \frac{\partial V}{\partial t} = -\frac{\partial V}{\partial T}
\]

**BSM Formula:**
- Call: \(\Theta_{call} = -\frac{S_0 n(d_1) \sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2)\)
- Put: \(\Theta_{put} = -\frac{S_0 n(d_1) \sigma}{2\sqrt{T}} + r K e^{-rT} N(-d_2)\)

**Interpretation:**
- Usually negative (options lose value as time passes)
- Expressed per day or per year
- Accelerates as expiration approaches

---

## 5. Finite Difference Method for Greeks

Since we use Monte Carlo (not analytical formulas), we compute Greeks using finite differences.

### Forward Difference (Delta, Vega, Rho)

\[
\frac{\partial V}{\partial x} \approx \frac{V(x + \epsilon) - V(x)}{\epsilon}
\]

**Delta:**
\[
\Delta \approx \frac{V(S_0 + \epsilon_S) - V(S_0)}{\epsilon_S}
\]

Where \(\epsilon_S\) is a small perturbation in stock price (typically 1% of \(S_0\)).

**Vega:**
\[
\nu \approx \frac{V(\sigma + \epsilon_\sigma) - V(\sigma)}{\epsilon_\sigma}
\]

Where \(\epsilon_\sigma\) is a small perturbation in volatility (typically 1% of \(\sigma\)).

**Rho:**
\[
\rho \approx \frac{V(r + \epsilon_r) - V(r)}{\epsilon_r}
\]

Where \(\epsilon_r\) is a small perturbation in rate (typically 0.1% = 0.001).

### Central Difference (Gamma)

\[
\Gamma \approx \frac{V(S_0 + \epsilon_S) - 2V(S_0) + V(S_0 - \epsilon_S)}{\epsilon_S^2}
\]

This is more accurate than forward difference for second derivatives.

### Theta (Time Decay)

\[
\Theta \approx \frac{V(T - \epsilon_T) - V(T)}{\epsilon_T}
\]

Where \(\epsilon_T\) is typically 1 day = \(1/252\) years (one trading day).

**Note:** Theta is usually negative, so we compute price after time passes forward.

### Choice of Epsilon (\(\epsilon\))

**Critical for accuracy:**

1. **Too small**: Numerical noise dominates (rounding errors)
2. **Too large**: Finite difference approximation breaks down
3. **Optimal**: 
   - **Delta/Gamma**: 1% of \(S_0\) (percentage-based)
   - **Vega**: 1% of \(\sigma\) (percentage-based)
   - **Rho**: 0.1% = 0.001 (absolute, since rates are small)
   - **Theta**: 1 day = 1/252 years

**Why percentage-based for Delta/Gamma/Vega?**
- Scales with the parameter value
- Works for any stock price/volatility level
- Maintains consistent relative accuracy

---

## 6. Common Random Numbers (CRN)

### Problem with Standard Monte Carlo Greeks

If we use different random numbers for \(V(S_0)\) and \(V(S_0 + \epsilon)\):

\[
\Delta = \frac{V(S_0 + \epsilon, Z_1) - V(S_0, Z_2)}{\epsilon}
\]

The difference includes:
1. **True sensitivity** (what we want)
2. **Monte Carlo noise** from different random numbers (what we don't want)

### Solution: Common Random Numbers

Use the **same** random numbers for both calculations:

\[
\Delta = \frac{V(S_0 + \epsilon, Z) - V(S_0, Z)}{\epsilon}
\]

Now the difference only reflects parameter sensitivity, not random variation.

### Implementation

1. Generate random matrix \(Z\) once: \(Z \sim \mathcal{N}(0,1)\) of size \((M, N)\)
2. For base price: \(V(S_0) = f(S_0, Z)\)
3. For bumped price: \(V(S_0 + \epsilon) = f(S_0 + \epsilon, Z)\)
4. Same \(Z\) used in both → noise cancels out

### Variance Reduction

CRN dramatically reduces variance of Greek estimates:

\[
\text{Var}[\Delta_{CRN}] \ll \text{Var}[\Delta_{standard}]
\]

**Intuition:** The correlation between \(V(S_0, Z)\) and \(V(S_0 + \epsilon, Z)\) is high, so their difference has low variance.

---

## 7. Stress Testing & Scenario Analysis

### Concept

Stress testing evaluates how option value and Greeks change under adverse market conditions.

### Mathematical Framework

Given base parameters \(\theta_0 = (S_0, \sigma_0, r_0, T_0)\), we compute:

\[
V(\theta_0) = \text{Base option price}
\]

\[
G(\theta_0) = \text{Base Greeks}
\]

Under stressed scenario \(\theta_{stress} = (S_0 + \Delta S, \sigma_0 + \Delta \sigma, r_0 + \Delta r, T_0 + \Delta T)\):

\[
V(\theta_{stress}) = \text{Stressed option price}
\]

\[
G(\theta_{stress}) = \text{Stressed Greeks}
\]

### Impact Measures

**Price Impact:**
\[
\Delta V = V(\theta_{stress}) - V(\theta_0)
\]

**Greek Sensitivity:**
\[
\Delta G = G(\theta_{stress}) - G(\theta_0)
\]

**Percentage Change:**
\[
\%\Delta G = \frac{G(\theta_{stress}) - G(\theta_0)}{|G(\theta_0)|} \times 100\%
\]

### Common Stress Scenarios

1. **Market Crash**: \(S_0 \to S_0 \times (1 - 0.05)\) (5% drop)
2. **Volatility Spike**: \(\sigma \to \sigma + 0.10\) (vol jumps 10 points)
3. **Rate Shock**: \(r \to r + 0.01\) (100 bps increase)
4. **Time Decay**: \(T \to T - 7/365\) (1 week passes)

### Why Traders Care

- **Risk Management**: Understand exposure to market moves
- **Hedging**: Determine hedge ratios under stress
- **Portfolio Analysis**: Aggregate risk across positions
- **Regulatory**: Stress testing requirements (e.g., VaR)

---

## 8. Exotic Options Payoffs

### Vanilla Option

**Call:**
\[
Payoff = \max(S_T - K, 0)
\]

**Put:**
\[
Payoff = \max(K - S_T, 0)
\]

### Asian Option

**Average Price Call:**
\[
Payoff = \max\left(\frac{1}{N}\sum_{i=1}^{N} S_{t_i} - K, 0\right)
\]

Where \(S_{t_i}\) are stock prices at observation dates.

**Key Feature:** Lower volatility than vanilla (averaging reduces variance)

### Barrier Option

**Down-and-Out Call:**
\[
Payoff = \begin{cases}
\max(S_T - K, 0) & \text{if } \min_{0 \leq t \leq T} S_t > B \\
0 & \text{if barrier } B \text{ is hit}
\end{cases}
\]

**Down-and-In Call:**
\[
Payoff = \begin{cases}
0 & \text{if barrier never hit} \\
\max(S_T - K, 0) & \text{if barrier } B \text{ is hit}
\end{cases}
\]

**Key Feature:** Path-dependent (depends on entire price path, not just \(S_T\))

### Lookback Option

**Call (strike = minimum):**
\[
Payoff = \max(S_T - \min_{0 \leq t \leq T} S_t, 0)
\]

**Put (strike = maximum):**
\[
Payoff = \max(\max_{0 \leq t \leq T} S_t - S_T, 0)
\]

**Key Feature:** Always in-the-money at expiration (best possible outcome)

---

## 9. Implementation Details

### Monte Carlo Path Generation

```python
# For each path i:
Z = np.random.standard_normal((paths, steps))  # Random shocks
drift = (r - 0.5*sigma^2) * dt                 # Risk-neutral drift
diffusion = sigma * sqrt(dt) * Z               # Volatility component
log_returns = drift + diffusion                # Log returns
cumulative_returns = cumsum(log_returns)       # Cumulative log returns
paths = S0 * exp(cumulative_returns)          # Stock price paths
```

### Greeks Calculation with CRN

```python
# Generate random matrix once
Z = np.random.standard_normal((paths, steps))

# Base price
V_base = price(S0, sigma, r, T, Z)

# Delta: bump S0
V_up = price(S0 + epsilon_S, sigma, r, T, Z)
Delta = (V_up - V_base) / epsilon_S

# Gamma: central difference
V_down = price(S0 - epsilon_S, sigma, r, T, Z)
Gamma = (V_up - 2*V_base + V_down) / epsilon_S^2

# Vega: bump sigma
V_vega = price(S0, sigma + epsilon_sigma, r, T, Z)
Vega = (V_vega - V_base) / epsilon_sigma
```

### BSM vs Monte Carlo Comparison

For vanilla options:
- **BSM**: Analytical, exact (under BSM assumptions)
- **Monte Carlo**: Numerical, approximate but flexible

**Validation:**
\[
\text{Error} = |V_{MC} - V_{BSM}|
\]

Should be small (typically < 0.1% for 10,000+ paths)

---

## 10. Interview Key Points

### GBM Questions

**Q: Why \((r - \sigma^2/2)\) in the drift?**
A: Under risk-neutral measure, expected return must equal risk-free rate. For log-normal process, \(E[\ln(S_T/S_0)] = (r - \sigma^2/2)T\) ensures \(E[S_T] = S_0 e^{rT}\).

**Q: What if volatility is stochastic?**
A: Need more complex models (Heston, SABR). GBM assumes constant volatility.

### Greeks Questions

**Q: Why is Gamma always positive?**
A: Options are convex in \(S\). As \(S\) increases, Delta increases (for calls) or becomes less negative (for puts).

**Q: How do you hedge Delta?**
A: Long \(\Delta\) shares per short option. Rebalance as Delta changes (Gamma risk).

**Q: What's the relationship between Delta and probability?**
A: For calls, \(\Delta \approx P(S_T > K)\) under risk-neutral measure. Not true probability, but risk-neutral probability.

### Monte Carlo Questions

**Q: How do you reduce variance?**
A: 
- Antithetic variates
- Common random numbers (for Greeks)
- Control variates
- Importance sampling
- More paths (but expensive)

**Q: When is Monte Carlo preferred over analytical methods?**
A: 
- Path-dependent options (barriers, Asians, lookbacks)
- Multiple underlying assets
- Complex payoffs
- American options (early exercise)

### Stress Testing Questions

**Q: What's the difference between stress testing and Greeks?**
A: 
- **Greeks**: Local sensitivity (small changes, first derivatives)
- **Stress testing**: Large moves, non-linear effects, tail risk

**Q: Why stress test if we have Greeks?**
A: Greeks assume small changes. Large moves can have non-linear effects (especially for exotic options).

---

## 11. Mathematical Relationships

### Put-Call Parity

\[
C - P = S_0 - K e^{-rT}
\]

**Implication:** If you know call price, you can derive put price (and vice versa).

### Greeks Relationships

**Delta-Gamma Relationship:**
\[
\Delta(S + dS) \approx \Delta(S) + \Gamma(S) \cdot dS
\]

**Theta-Gamma Relationship (for delta-neutral portfolio):**
\[
\Theta \approx -\frac{1}{2}\sigma^2 S^2 \Gamma
\]

This is the "theta-gamma trade-off" in delta hedging.

### Risk-Neutral Pricing

Under risk-neutral measure \(\mathbb{Q}\):

\[
V_0 = e^{-rT} \mathbb{E}^Q[Payoff]
\]

All assets earn risk-free rate \(r\) (not the actual expected return \(\mu\)).

---

## 12. Numerical Considerations

### Time Discretization Error

Using \(N\) steps with \(\Delta t = T/N\):

\[
\text{Error} = O(\Delta t) = O(1/N)
\]

More steps → smaller error, but more computation.

### Monte Carlo Error

With \(M\) paths:

\[
\text{Standard Error} = \frac{\sigma}{\sqrt{M}} = O(1/\sqrt{M})
\]

To halve error, need 4× paths.

### Greeks Accuracy

Using finite differences with \(\epsilon\):

\[
\text{Truncation Error} = O(\epsilon) \text{ (forward difference)}
\]

\[
\text{Truncation Error} = O(\epsilon^2) \text{ (central difference)}
\]

But smaller \(\epsilon\) → more numerical noise. Optimal \(\epsilon\) balances these.

---

---

## 13. Practical Examples & Interview Scenarios

### Example 1: Calculating Delta

**Setup:**
- Stock: \(S_0 = \$100\)
- Strike: \(K = \$100\)
- Volatility: \(\sigma = 20\%\)
- Rate: \(r = 5\%\)
- Time: \(T = 1\) year

**Monte Carlo Approach:**
1. Generate paths with \(S_0 = 100\), get price \(V_0 = \$10.50\)
2. Generate paths with \(S_0 = 101\) (1% bump), get price \(V_1 = \$10.65\)
3. Delta = \((10.65 - 10.50) / 1 = 0.15\)

**BSM Approach:**
- \(d_1 = 0.325\), \(N(d_1) = 0.627\)
- Delta = 0.627

**Interpretation:** For every \$1 increase in stock price, call option increases by \$0.63.

### Example 2: Stress Test Scenario

**Base Scenario:**
- \(S_0 = \$100\), \(\sigma = 20\%\), \(r = 5\%\), \(T = 1\) year
- Call option price: \$10.50
- Delta: 0.63

**Stress: Market Crash (-10%)**
- \(S_0 \to \$90\)
- New price: \$4.20
- New Delta: 0.35
- **Impact:** Price drops \$6.30 (60%), Delta drops 0.28 (hedge ratio changes!)

**Key Insight:** Greeks change under stress - static Greeks don't tell full story.

### Example 3: Why CRN Matters

**Without CRN:**
- \(V(S_0 = 100, Z_1) = 10.50\)
- \(V(S_0 = 101, Z_2) = 10.48\) (different random numbers!)
- Delta = \((10.48 - 10.50)/1 = -0.02\) ❌ **Wrong sign!**

**With CRN:**
- \(V(S_0 = 100, Z) = 10.50\)
- \(V(S_0 = 101, Z) = 10.65\) (same random numbers)
- Delta = \((10.65 - 10.50)/1 = 0.15\) ✅ **Correct!**

### Example 4: Gamma Hedging

**Problem:** Delta hedge needs constant rebalancing due to Gamma.

**Solution:** Create delta-gamma neutral portfolio:
- Short 1 call option (\(\Delta = 0.6\), \(\Gamma = 0.05\))
- Long 0.6 shares (\(\Delta = 0.6\), \(\Gamma = 0\))
- Long \(x\) units of another option with \(\Gamma_2 = 0.03\)

**Gamma neutral condition:**
\[
-0.05 + x \cdot 0.03 = 0 \implies x = 1.67
\]

Now portfolio is both delta and gamma neutral (less rebalancing needed).

---

## 14. Common Interview Questions & Answers

### Q1: Explain the risk-neutral measure

**Answer:**
- In risk-neutral world, all assets earn risk-free rate \(r\)
- We don't use actual expected return \(\mu\)
- Option prices = discounted expected payoff under \(\mathbb{Q}\)
- GBM drift becomes \((r - \sigma^2/2)\) instead of \((\mu - \sigma^2/2)\)
- This allows pricing without knowing investor risk preferences

### Q2: Why do we use \((r - \sigma^2/2)\) instead of just \(r\)?

**Answer:**
- For log-normal process: \(E[\ln(S_T/S_0)] = (\mu - \sigma^2/2)T\)
- But \(E[S_T] = S_0 e^{\mu T}\) (Jensen's inequality)
- Under risk-neutral: \(E[S_T] = S_0 e^{rT}\)
- So we need \(E[\ln(S_T/S_0)] = (r - \sigma^2/2)T\) to ensure this

### Q3: What's the difference between historical and implied volatility?

**Answer:**
- **Historical**: Calculated from past stock returns
- **Implied**: Backed out from market option prices using BSM
- Traders use implied vol (forward-looking, market consensus)
- Historical vol used for risk management, backtesting

### Q4: How do you validate Monte Carlo results?

**Answer:**
1. **BSM comparison** (for vanilla options)
2. **Convergence test**: Increase paths, check if estimate stabilizes
3. **Standard error**: Should decrease as \(1/\sqrt{M}\)
4. **Put-call parity**: Check \(C - P = S - Ke^{-rT}\)
5. **Boundary conditions**: Check at \(T=0\), deep ITM/OTM

### Q5: Explain the Greeks in trading context

**Answer:**
- **Delta**: How many shares to hedge (hedge ratio)
- **Gamma**: How often to rebalance delta hedge
- **Vega**: Exposure to volatility changes (vol trading)
- **Rho**: Interest rate risk (less important for short-dated)
- **Theta**: Daily P&L from time decay (theta decay)

### Q6: What happens to Greeks as expiration approaches?

**Answer:**
- **Delta**: Approaches 0 or 1 (binary outcome)
- **Gamma**: Explodes near expiration (high sensitivity)
- **Vega**: Decreases (less time for vol to matter)
- **Theta**: Increases (accelerated time decay)

### Q7: Why use Monte Carlo for exotics?

**Answer:**
- **Analytical formulas**: Don't exist for most exotics
- **Path-dependence**: Need full price path (barriers, Asians)
- **Flexibility**: Easy to add features (early exercise, dividends)
- **Multiple assets**: Extends naturally to basket options

### Q8: How do you choose number of paths and steps?

**Answer:**
- **Paths**: More = lower error, but \(O(1/\sqrt{M})\) convergence
  - Typical: 10,000-100,000 for pricing
  - 5,000-10,000 for stress tests (faster)
- **Steps**: More = better path resolution
  - Daily pricing: steps = trading days
  - Weekly: steps = weeks to expiration
  - Trade-off: More steps = more computation

---

## 15. Code Implementation Mapping

### GBM Implementation
```python
# stochastic_process.py
drift = (r - 0.5 * sigma**2) * dt
diffusion = sigma * sqrt(dt) * Z
log_returns = drift + diffusion
paths = s0 * exp(cumsum(log_returns))
```
**Formula:** \(S_t = S_0 \exp[(r - \sigma^2/2)t + \sigma \sqrt{t} Z]\)

### Greeks with CRN
```python
# greek_calculator.py
Z = random_matrix  # Generated once
V_base = price(S0, Z)
V_up = price(S0 + ds, Z)  # Same Z!
Delta = (V_up - V_base) / ds
```
**Formula:** \(\Delta = \frac{V(S+\epsilon) - V(S)}{\epsilon}\) with same \(Z\)

### BSM Pricing
```python
# bsm_pricer.py
d1 = (log(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
C = S*N(d1) - K*exp(-r*T)*N(d2)
```
**Formula:** Standard BSM call formula

---

## Summary

This pricer implements:
1. **GBM** for stock price evolution
2. **Monte Carlo** for flexible option pricing
3. **Finite differences** with **CRN** for accurate Greeks
4. **BSM** for vanilla option benchmarking
5. **Stress testing** for risk analysis

All methods are mathematically rigorous and industry-standard for options pricing and risk management.

### Key Takeaways for Interviews

1. **Risk-neutral pricing**: Use \(r\), not \(\mu\)
2. **CRN is critical**: Without it, Greek estimates are noisy
3. **Epsilon choice matters**: Percentage-based for Delta/Gamma/Vega
4. **Stress testing complements Greeks**: Shows non-linear effects
5. **Monte Carlo is flexible**: Works for any payoff structure
6. **BSM validates MC**: Should match closely for vanilla options
