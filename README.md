# Neural ODE and ResNet toolbox

Small toolbox for solving supervised learning tasks using neural ODEs and ResNets.
The code has been used to generate figures which appear in several papers.

These include:

1. "**Large-time asymptotics in deep learning**" by Carlos Esteve-Yague, Borjan Geshkovski, Dario Pighin and Enrique Zuazua.

```
@article{esteve2021large,
  title={Large-time asymptotics in deep learning},
  author={Esteve-Yag{\"u}e, Carlos and Geshkovski, Borjan and Pighin, Dario and Zuazua, Enrique},
  year={2021}
}
```

2. "**Sparse approximation in learning via neural ODEs**" by Carlos Esteve-Yague and Borjan Geshkovski.

```
@article{esteve2021sparse,
  title={Sparse approximation in learning via neural ODEs},
  author={Esteve-Yag{\"u}e, Carlos and Geshkovski, Borjan},
  journal={arXiv preprint arXiv:2102.13566},
  year={2021}
}
```

3. "**Turnpike in Lipschitz-nonlinear optimal control**" by Carlos Esteve-Yague, Borjan Geshkovski, Dario Pighin and Enrique Zuazua. 

```
@article{esteve2020turnpike,
  title={Turnpike in Lipschitz-nonlinear optimal control},
  author={Esteve-Yag{\"u}e, Carlos and Geshkovski, Borjan and Pighin, Dario and Zuazua, Enrique},
  journal={arXiv preprint arXiv:2011.11091},
  year={2020}
}
```

![Comparison of ISTA and NonCvx-Pro flows](imgs/flows.png)

The toolbox can be improved by further adding the following functionalities: 
- Weight clipping for more complex networks to ensure $L^1$ and $L^\infty$ constraints.
- Time-dependent weights for non-uniform time-stepping.
