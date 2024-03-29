<!-- Title -->
<h1 align="center">
  Leia: Learning with EvolutIon equAtions
</h1>

<!-- Information badges -->
<p align="center">
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dedalus">
  <a href="https://www.repostatus.org/#concept">
    <img alt="Repo status" src="https://www.repostatus.org/badges/latest/concept.svg" />
  </a>
  <a href="https://doi.org/10.1017/S0962492922000046">
  <img src="https://zenodo.org/badge/DOI/10.1017/S0962492922000046.svg">
  </a>
</p>

<p align="center">
  <img src="videos/traj.gif" alt="animated" width="302", height="311"/>
  &nbsp;
  &nbsp;
  &nbsp;
  <img src="videos/gen.gif" alt="animated" width="365", height="311"/>
</p>

A <tt>PyTorch</tt> toolbox for solving learning tasks with neural ODEs. An important element of this toolbox is that it allows for time-dependent weights (controls), and costs involving integrals of the state. Generally speaking, there is flexibility in using different functionals and weight penalties (beyond simply $L^2$, for instance).

A sample experiment may be found in <tt>generate_fig.py</tt>, with the main modules being a simple instantiation of the neural ODE

```python
model = NeuralODE(device, 
                  data_dim=2, 
                  hidden_dim=5, 
                  augment_dim=1, 
                  non_linearity='relu',
                  architecture='bottleneck', 
                  T=10, 
                  time_steps=20, 
                  fixed_projector=False, 
                  cross_entropy=False)
```
and then of the optimization algorithm

```python
trainer = Trainer(model, 
                        optimizer_anode, 
                        device, 
                        cross_entropy=False, 
                        turnpike=True,
                        bound=0., 
                        fixed_projector=False)
```

## Citing 

If you are using this toolbox for your scientific publication, we would be very appreciative if you were to cite one of our following articles on this topic.

1. **Turnpike in optimal control of PDEs, ResNets, and beyond**

```bibtex
@article{geshkovski2022turnpike,
  title={Turnpike in optimal control of PDEs, ResNets, and beyond},
  author={Geshkovski, Borjan and Zuazua, Enrique},
  journal={Acta Numerica},
  volume={31},
  pages={135--263},
  year={2022},
  publisher={Cambridge University Press}
}
```

2. **Large-time asymptotics in deep learning**

```bibtex
@article{esteve2021large,
  title={Large-time asymptotics in deep learning},
  author={Esteve-Yag{\"u}e, Carlos and Geshkovski, Borjan and Pighin, Dario and Zuazua, Enrique},
  year={2021}
}
```

3. **Sparse approximation in learning via neural ODEs**

```bibtex
@article{esteve2023sparsity,
  title={Sparsity in long-time control of neural ODEs},
  author={Esteve-Yag{\"u}e, Carlos and Geshkovski, Borjan},
  journal={Systems \& Control Letters},
  volume={172},
  pages={105452},
  year={2023},
  publisher={Elsevier}
}
```

<!-- ## Improvements

The toolbox can be improved by further adding the following functionalities: 
- Weight clipping for bottleneck architectures to ensure $L^1$ and $L^\infty$ constraints.
- Time-dependent weights for non-uniform time-stepping. -->
