# Galaxy Builder rendering 

Boxy ellipse calculation in galaxy builder is done by

$$ D = 3{R_e}^{-\frac{c}{2}}\sqrt{\left(a|x'|\right)^c + |y'|^c}$$

where $a$ is the axis ratio, $R_e$ is scale, $c$ is the boxyness modifier and

$$
\begin{pmatrix} x' \\\\ y' \end{pmatrix} 
= \begin{pmatrix}
    \cos\theta       & -\sin\theta \\\\
    \sin\theta       &  \cos\theta \\\\
\end{pmatrix}\begin{pmatrix} x \\\\ y \end{pmatrix} 
- \begin{pmatrix}
    \cos\theta       & -\sin\theta \\\\
    \sin\theta       &  \cos\theta \\\\
\end{pmatrix}\overrightarrow\mu
$$

rotates the isophote.

The Sérsic rendering function for galaxy builder is given by

$$\Sigma(r ) = \frac{i_0}{2} \exp\left[-b_n\left(D^{1 / n} - 1\right)\right]$$

From [the galfit paper](https://arxiv.org/pdf/0912.0731.pdf), the equation for a boxy ellipse distance is 

$$r(x,y) = \left(|x - \mu_x|^{C_0 + 2} + \left|\frac{y - \mu_y}{q}\right|^{C_0+2}\right)^{\frac{1}{C_0 + 2}},$$

and the Sersic rendering function is

$$\Sigma(r ) = \Sigma_e\exp\left[-\kappa\left(\left(\frac{r}{r_e}\right)^{1/n} - 1\right)\right]$$

So we can see that I messed up and D should have a power of $1/c$ not a square root, which would mean $r_e = R_e$. With the above model, however, this is not the case.

$$\Sigma_e = \frac{i_0}{2};\; \kappa = b_n;\; D = \left(\frac{r}{r_e}\right)$$


## Total flux calculation

The galfit paper gives

$$ F_{\text{tot}} = 2\pi r_e\Sigma_e e^\kappa n \kappa^{-2n}\Gamma(2n)q/R(C_0;\ m).$$

Where $R(C_0;\ m)$ is a geometric correction term for when the isophote deviates from a perfect ellipse.

For a boxyness modifier

$$R(C_0;\ m) = \frac{\pi C_0}{4\beta(C_0^{-1},\ 1 + C_0^{-1})},$$

where $\beta$ is the Beta function and $\Gamma$ is the Gamma function.

