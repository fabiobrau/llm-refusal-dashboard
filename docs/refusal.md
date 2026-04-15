# Refusal Suppression.

The refusal suppression is called SD (arditi et al.) and is the simples (also the first) ever invented.

The idea is to select the centroids \mu and \nu of harmful and harmless latent-states at the post-instruction token at a given layer l*.

The refusal direction is then computed by r = \mu - \nu


The steering of all the generated tokens after the post-instruction token is then computed by considering 

h = h - \beta rr^Th / || r||^2

where \beta is an impact factor from 0 to some upper bound B

The steering is applied with the same r to all the layers. This is the idea of Arditi method.