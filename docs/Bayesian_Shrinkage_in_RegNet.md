# Bayesian Shrinkage in RegNet

**Last updated:** 2025-05-24

This note formalises the Bayesian interpretation of RegNet's first
GraphSAGE layer and explains how we compute the *effective* shrinkage
weight that appears in the diagnostic plots
(`shrinkage_vs_degree.png`, `shrinkage_hist.png`, `shrinkage_vs_expr.png`).

---
## 1 Statistical model

For one gene *i* in one cell *s*

| quantity | distribution | description |
|-----------|--------------|-------------|
| expression | \(x_{is}\,\mid\,\theta_{is} \sim \mathcal N(\theta_{is},\,\sigma^2)\) | technical noise |
| prior | \(\theta_{is}\,\mid\,m_{is} \sim \mathcal N(m_{is},\,\tau_i^2)\) | neighbour mean \(m_{is}=\frac1{|\Gamma(i)|}\sum_{j\in\Gamma(i)}x_{js}\) |

With precisions \(\lambda=\sigma^{-2}\) and \(\rho_i=\tau_i^{-2}\)
Bayes' rule gives the posterior mean

\[
 \mu_{is}^* \;=\;(1-w_{is})\,m_{is} \;+\; w_{is}\,x_{is},
\quad
 w_{is}= \frac{\lambda}{\lambda+\rho_i} \,\in (0,1).\tag{1}
\]

Vectorising across the *S* cells yields

\[ \boldsymbol\mu_i^* = (1-w_i)\,\boldsymbol m_i + w_i\,\boldsymbol x_i. \]

---
## 2 Idealised linear GraphSAGE encoder

If the first layer were strictly

\[
 h_i = W_\mathrm{self} \, x_i + W_\mathrm{neigh} \, m_i \quad (d_h\times F)\tag{2}
\]

and the two weight matrices were close to multiples of the identity
\(W_\mathrm{self}\approx w_i I,\; W_\mathrm{neigh}\approx(1-w_i) I\),
then *every* feature of \(h_i\) would realise Eq. (1).

---
## 3 Actual implementation (MLP aggregator)

```python
concat = torch.cat([x_i, m_i], dim=-1)   # 2F-vector
h_i    = W2 @ ReLU( W1 @ concat + b1 ) + b2    # ℝ^{d_h}
```

Because of the ReLU non-linearity and the change of feature space,
(1) no longer holds exactly.

However, ReLU is *piece-wise linear*.  For a fixed activation pattern
around \((x_i, m_i)\) we have the *exact* local representation

\[ h_i = A_i x_i + B_i m_i + b_i, \]    
with \(A_i, B_i \in \mathbb R^{d_h\times F}\).

---
## 4 Directional sensitivity

Let \(\delta_i = x_i - m_i\) (vector pointing from prior to data).
Define two local sensitivities

\[
\begin{aligned}
S_{x,i} &= \bigl\| A_i\,\delta_i \bigr\|_2,\\[4pt]
S_{m,i} &= \bigl\| B_i\,\delta_i \bigr\|_2.
\end{aligned}\tag{3}
\]

If the layer were linear with form (2) these would satisfy
\(S_{x,i}:S_{m,i}=w_i:1\! -\! w_i\).
Hence we *define*

\[
\boxed{\displaystyle
w_i^{\mathrm{eff}} = \frac{S_{x,i}}{S_{x,i}+S_{m,i}}}
\qquad (0 \le w_i^{\mathrm{eff}} \le 1).\tag{4}
\]

---
## 5 Finite-difference approximation

In practice we estimate (3) by a centred finite difference with step
\(\varepsilon\approx10^{-2}\) along the unit vector
\(\hat\delta_i = \delta_i/\|\delta_i\|_2\):

\[
\begin{aligned}
S_{x,i} &\approx \frac{\| g(x_i+\varepsilon\hat\delta_i, m_i) - g(x_i,m_i) \|_2}{\varepsilon},\\[4pt]
S_{m,i} &\approx \frac{\| g(x_i, m_i+\varepsilon\hat\delta_i) - g(x_i,m_i) \|_2}{\varepsilon}.
\end{aligned}\tag{5}
\]

Inside a ReLU region Eq. (5) is **exact** because the Jacobian is
constant.

---
## 6 When does \(w_i^{\mathrm{eff}}\) equal the analytic weight?

*If and only if* the aggregator is linear of form (2) with matrices
proportional to the identity.  In that special case

\[ w_i^{\mathrm{eff}} \equiv w_i = \frac{\sigma^2}{\sigma^2+\tau_i^2}. \]

For non-linear networks \(w_i^{\mathrm{eff}}\) generalises the precision
ratio by comparing *directional* derivatives instead of scalar weights.

---
## 7 Practical implications

* **Positive trend with in-degree.** As \(|\Gamma(i)|\uparrow\) the
  empirical prior variance \(\tau_i^2\downarrow\Rightarrow S_{m,i}\downarrow\)
  so \(w_i^{\mathrm{eff}}\uparrow\).

* **Negative trend with expression.** Genes with high mean expression
  have lower measurement noise \(\sigma^2\), pushing
  \(w_i^{\mathrm{eff}}\downarrow\).

* **TF vs non-TF.** We observe systematically lower
  \(w_i^{\mathrm{eff}}\) for TFs, indicating their embeddings conserve
  more self-specific information.

---
## 8 Reproducing the computation

```python
# x, m : (N,F) tensors, model : trained RegNet
first_layer = model.graphsage.layers[0].agg

delta = x - m
unit  = delta / delta.norm(dim=1, keepdim=True).clamp(min=1e-9)

eps   = 1e-2
out0  = first_layer(x, m)
Sx    = (first_layer(x + eps*unit, m) - out0).norm(dim=1) / eps
Sm    = (first_layer(x, m + eps*unit) - out0).norm(dim=1) / eps
w_eff = Sx / (Sx + Sm)
```

The resulting `w_eff` is exactly what you see in the shrinkage plots in
`pretrain_outputs/<dataset>/pretrain/`.

---
## 9 Limitations

1. Crossing a ReLU boundary within \(\varepsilon\) causes a slight
   under-estimation of \(S\).  Using a smaller \(\varepsilon\) and
   batching mitigates this.
2. The derivation treats every gene independently; higher-layer effects
   (attention, VAE) are not captured.
3. GraphSAGE degree is computed on the **incoming** edge set; interpret
   plots accordingly.

---
## 10 References

* Robbins H. (1964) *The Empirical Bayes Approach …* Proc. 4th Berkeley
  Symp.
* Hamilton W. et al. (2017) *Inductive Representation Learning on
  Large Graphs.* NIPS.
* Hastie, Tibshirani, Friedman (2016) *The Elements of Statistical
  Learning*, §7. 