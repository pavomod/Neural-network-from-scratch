# Neural Network 🧠
### Progetto 📈
Questo progetto mira a sviluppare una Rete Neurale Artificiale da zero, utilizzando Python senza fare affidamento su librerie pre-esistenti. Il processo di validazione comprende la selezione di uno spazio di ricerca per gli iperparametri e l'esecuzione di una ricerca a griglia. Successivamente, si effettuano raffinamenti per valutare potenziali miglioramenti. Ogni modello viene convalidato utilizzando una validazione incrociata a 5 fold.

### Rete finale 🚀
La rete neurale finale è configurata con 4 strati nascosti come segue:

Primo strato: 20 nodi, funzione di attivazione tanH, inizializzazione He
Secondo strato: 30 nodi, funzione di attivazione tanH, inizializzazione He
Terzo strato: 20 nodi, funzione di attivazione tanH, inizializzazione He
Quarto strato: 30 nodi, funzione di attivazione tanH, inizializzazione He
Quinto strato: 25 nodi, funzione di attivazione ReLU, inizializzazione He
Strato di output: 3 nodi, funzione di attivazione lineare, inizializzazione uniforme
