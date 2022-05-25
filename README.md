# Deep Latent Regularity Network for Modeling Stochastic Partial Differential Equations

**Abstract**: Stochastic partial differential equations (SPDEs) are crucial for modeling dynamics with randomness in many areas including economics, physics, and atmospheric sciences. Recently, using data-driven methods to learn the PDE solution for accelerating PDE simulation becomes increasingly popular. However, learning a well-generalized SPDE solution is still challenging due to the poor regularity \footnote{Roughly speaking, regularity describes the smoothness of a function.} of the random forcing. In this work, we incorporate physics-informed features inspired by the regularity structure theory with deep neural network to model the SPDEs' mild solution. We propose \emph{Deep Latent Regularity Net} (DLR-Net), which maps the initial condition and random forcing to the SPDE's mild solution. DLR-Net includes regularity feature block as a main component, which consecutively encodes the random forcing to regularity features by kernel smoothing. Specifically, the kernel is designed according to the linear part of the SPDE, which is physics-informed and there's no learnable weights. We conduct experiments on various SPDEs including the dynamic $\Phi^4_1$ model and the stochastic 2D Navier-Stokes equation to predict their solutions, and the results demonstrate that the proposed DLR-Net can achieve one order of magnitude lower prediction error. The inference time is over 20 times faster than traditional numerical solver and is comparable with the baseline deep learning models.

---
## Environment
Our codes are run on Linux system with pytorch 1.10.2 and cuda 11.3. Run following code to create a conda environment `DLRNet`.
```bash
sh env.sh
```
## Run experiments

### Dynamic $\Phi^4_1$ Model
<!-- $$
    \begin{aligned}
    \partial_t u-\Delta u &= 3u-u^3+\xi,\quad(t,x)\in[0,T] \times D\\
    u(t,0) &= u(t,1),\quad\rm(Periodic\ BC)\\
    u_0(x) &= u(0,x) = x(1-x) + \kappa\eta(x),
    \end{aligned}
$$ -->
- Data Generation:

    To generate all data, run
    ```
    sh phi41_data_gen.sh
    ```
     or use following code
    ```bash
    python phi41_data.py -N 1000 -k 0.0
    ```
    to generate one dataset.

- Model training:
    ```bash
    python phi41.py -N 1000 -k 0.0
    ```


### Reaction-Diffusion Equation with Linear Multiplicative Forcing

- Data Generation:

    To generate all data, run
    ```
    sh mult_data_gen.sh
    ```
     or use following code
    ```bash
    python mult_data.py -N 1000 -k 0.0
    ```
    to generate one dataset.

- Model training:
    ```bash
    python mult.py -N 1000 -k 0.0
    ```

### Stochastic 2D Navier-Stokes Equation

- Dataset: We use datasets published by [Neural SPDEs](https://github.com/crispitagorico/Neural-SPDEs). Specifically, we use [NS_xi.mat](https://osf.io/ahn6v/files/googledrive/NS_xi.mat/) and [NS_u0_xi.mat](https://osf.io/ahn6v/files/googledrive/NS_u0_xi.mat/) to train and evaluate our model.

- Model training:
    Use
    ```bash
        python NS.py
    ```
    to train and evalute $(w_0, \xi) \mapsto w$ setting. Use
    ```bash
        python NS.py --fixU0
    ```
    to train and evalute $\xi \mapsto w$ setting.

---

## Acknowledgements

Some codes for numerical simulations in [Feature Engineering with Regularity Structures](https://github.com/andrisger/Feature-Engineering-with-Regularity-Structures) and [Neural-SPDEs](https://github.com/crispitagorico/Neural-SPDEs) are referenced to generate training datasets and regularity features. [Fourier Neural Operator](https://github.com/zongyi-li/fourier_neural_operator) is also referenced in constructing the decoder layers. 
