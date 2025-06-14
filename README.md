El codigo implementa un modelo de transferencia en régimen transitorio. Utiliza una matriz de masa lumpeada en el método EulerForward (run_transient_explicit) y una matriz de masa consistente en el EulerBackward (run_transient_implicit). 
La solución mediante FEM puede ser graficada para cualquier combinación de condiciones de borde Dirichlet/Neumann. La solución analítica es graficada y validada con la solución FEM únicamente 
para combinaciones Dirichlet-Dirichlet, Dirichlet-Neumann y Neumann-Neumann, siempre y cuando las condiciones de borden impongan valor cero en ambos extremos. 

Para el método EulerForward se implementó a stable_dt la cual calcula el mínimo dt necesario para garantizar la estabilidad. En EulerBackward se mantuvo el dt utilizado en el código original provisto por la cátedra, pero
los resultados son satisfactorios para valores de dt mucho más grandes. 
