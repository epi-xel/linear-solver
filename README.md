# Introduzione

L'obiettivo del progetto qui presentato è l'implementazione di una
libreria che esegua i solutori iterativi di **Jacobi**,
**Gauss-Seidel**, **gradiente** e **gradiente coniugato**, ma che
fornisca anche una serie di *utility* per aiutare la raccolta di
statistiche, la creazione di grafici, la stampa a schermo e il
riutilizzo di codice.

La libreria è consultabile al repository
[`epi-xel/linear-solver`](https://github.com/epi-xel/linear-solver) su
GitHub.

# Tecnologie utilizzate

Si è scelto di utilizzare **Python** come linguaggio di programmazione
con uno scopo ben specifico. Infatti, data la flessibilità che offre
nell'implementazione software e l'ampia scelta di librerie per i grafici
e il calcolo scientifico, era uno dei pochi linguaggi che permettesse
uno sviluppo agile a 360 gradi con delle ottime librerie per la
produzione di grafici.

Le principali librerie di Python che sono state quindi utilizzate per
l'implementazione vengono descritte di seguito:

-   **Scipy** e **Cholmod**, per la gestione delle *matrici sparse*: il
    package `scipy.sparse` offre infatti un'ampia gamma di funzioni per
    trattare le matrici sparse, dalla lettura di file formato `.mtx`,
    all'esecuzione di operazioni base tra matrici; `sksparse.cholmod`
    permette invece di utilizzare il metodo di Choleski su matrici
    sparse.

-   **Pandas**, **Matplotlib** e **Seaborn**, per l'analisi dei
    risultati, produzione di tabelle riassuntive e grafici.

# Struttura della libreria

Il progetto si compone di due parti: una libreria e un eseguibile con
un'interfaccia da linea di comando.

## Libreria `linearsolver`

Data la complessità dell'idea progettuale, si è voluta strutturare la
libreria in tre distinti moduli principali, ciascuno di essi contenente
sottomoduli più specifici:

1.  **Helpers**: modulo dedicato alla definizione di classi di oggetti
    utilizzati per supportare/aiutare l'esecuzione di altri metodi

2.  **Methods**: modulo che implementa gli algoritmi veri e propri per
    la risoluzione dei sistemi lineari

3.  **Utils**: modulo adibito all'implementazione di funzioni di varia
    natura, come per la stampa a schermo dei risultati e l'analisi di
    questi ultimi

Si è cercato inoltre di coniugare la flessibilità di Python per il
calcolo numerico e l'implementazione di classi di oggetti focalizzati al
disaccoppiamento delle funzioni della libreria che seguissero i più
classici design patterns. L'obiettivo era di mantenere la semplicità
caratteristica di Python nelle funzioni base della libreria e
arricchendo organicamente il resto delle funzioni ausiliarie per creare
codice mantenibile, documentabile, facilmente utilizzabile ed incline a
più livelli di astrazione.

### Helpers

Tra i sottomoduli qui presenti, le classi di oggetti più interessanti
che vengono definite sono:

-   `LinearSystemHelper`: classe di oggetti che permette la
    memorizzazione di alcune variabili utili durante l'esecuzione dei
    metodi per la risoluzione dei sistemi, come la matrice $A$, il
    vettore $b$, il vettore soluzione $x$, il vettore soluzione esatta
    $x_{true}$ ed eventuali variabili specifiche per il metodo chiamato.

-   `LinearSystemResult`: classe di oggetti per l'incapsulamento dei
    risultati ottenuti dagli algoritmi di risoluzione; in particolar
    modo viene salvato il vettore soluzione, il tempo di esecuzione, il
    numero di iterazioni e l'errore relativo.

-   `ResultsStats`: classe di oggetti che si comporta similmente ad un
    *dataframe* per la memorizzazione delle statistiche sui risultati,
    contenente alcuni metodi che permettono l'aggiunta di ulteriori
    statistiche o operare la fusione con un altro oggetto della stessa
    classe. Segue il design pattern *builder*, dal momento che cerca di
    ottimizzare la costruzione iterativa del dataframe, che sarebbe
    altrimenti molto onerosa e si comporta anche da *mediator* tra i
    moduli per la produzione e l'elaborazione dei risultati.

### Methods

Volendo applicare il design pattern *stretegy*, questo modulo contiene
tre sottomoduli principali:

-   `base_solver`: data la struttura comune di ciascun metodo si sono
    raccolte in questo modulo le funzioni condivise dai metodi.

-   `update`: in questo modulo sono raccolte le funzioni di update
    distinte per ciascun metodo.

-   `methods_collecto`: il modulo funge, in termini di design patterns,
    da *facade*, raccogliendo una serie di funzioni ausiliarie che
    permettono l'esecuzione di tutti i metodi per un singolo input con
    una certa tolleranza e, utilizzando `print_utils` (cfr. sezione
    [3.1.3](#sec:utils))
    stampa a schermo i risultati.

Si approfondiscono di seguito i primi due moduli dedicati al cuore vero
e proprio del progetto.

#### `base_solver`

Il modulo si avvale di una funzione principale che contiene la struttura
base degli algoritmi iterativi: la funzione prende in ingresso $A$, $b$,
$x$, la tolleranza, il massimo numero di iterazioni e l'elemento di
un'enumerazione (sempre definita in questo stesso modulo), che definisce
il metodo che si desidera eseguire. Il metodo al suo interno seguendo
l'implementazione classica degli algoritmi iterativi, calcola quindi il
risultato, il tempo di esecuzione e il numero di iterazioni, servendosi
durante le iterazioni dell'oggetto della classe `LinearSystemHelper` e
incapsulando alla fine il risultato in un oggetto della classe
`LinearSystemResult`, che restituisce al chiamante.

All'interno del modulo sono anche definite due funzioni ausiliare, di
cui una calcola ad ogni iterazione la convergenza dell'algoritmo,
secondo la seguente formula:

$$\frac{\Vert Ax^{(k)}-b \Vert}{\Vert b \Vert} < \texttt{tol}$$

L'altra funzione invece viene eseguita prima di tutto per stabilire se
la matrice è simmetrica e definita positiva, attraverso l'esecuzione del
metodo di Choleski messo a disposizione dalla libreria `cholmod`:
eventuali eccezioni vengono catturate e il controllo viene restituito al
chiamante, stampando a schermo un messaggio di errore.

#### `update`

Questo modulo definisce le operazioni di update per ciascun metodo:
**Jacobi**, **Gauss-Seidel**, **gradiente** e **gradiente coniugato**.
Ciascuna delle funzioni principali accetta in ingresso un oggetto della
classe `LinearSystemHelper`, su cui effettuano le operazioni di modifica
e aggiornamento. Per ciascun metodo sono state utilizzate le operazioni
base tra matrici sparse offerte dal package `scipy.sparse`, mentre
relativamente a Gauss-Seidel si è implementata a parte nella libreria
una funzione che eseguisse la sostituzione in avanti, passo fondamentale
in ogni aggiornamento.

### Utils

I moduli di *utility* sono:

-   `analize`: modulo che implementa la realizzazione di grafici ed
    esportazione di tabelle sulla base dei risultati ottenuti dagli
    algoritmi che implementano i metodi. Per interfacciarsi al metodo
    principale di questo modulo, `export_results()`, di questo modulo è
    necessario utilizzare un oggetto della classe `ResultsStats` che
    incapsula i dati dei risultati e facilita la loro gestione tra le
    funzioni di questo modulo.

-   `print_utils`: modulo che raccoglie una serie di funzioni che
    permettono la stampa a schermo dei risultati ottenuti dai metodi.
    Per interfacciarsi al metodo principale di questo modulo si deve
    impiegare un oggetto della classe `LinearSystemResult`.

-   `constants`: modulo dove sono contenute alcune costanti utilizzate
    nell'intero programma.

## Eseguibile

Una volta implementata la libreria, si è voluto produrre un eseguibile
che avesse un'interfaccia intuitiva da linea di comando. Eseguendo il
file con il comando `–help` (oppure se non si inserisce nulla) si
ottiene l'output rappresentato nella figura
[1].

![Interfaccia da linea di comando del
programma.](images/program-cli.png)

Da linea di comando è pertanto possibile inserire una o più matrici con
diverse tolleranze e ottenere un output sia sul terminale, sia su file,
con alcuni grafici e file CSV che riassumono l'esecuzione dei metodi.

Per semplicità da linea di comando è possibile dare in input solo
matrici, dato che all'interno dell'eseguibile viene calcolato $Ax = b$
dove $x$ è un vettore di soli uno. All'interno del file è però possibile
sovrascrivere il vettore $x$ o $b$ con qualsiasi vettore desiderato.

Per ciascun sistema lineare in input viene calcolata la soluzione con
ciascuna tolleranza e ciascun metodo, avvalendosi del modulo `big_ops`
descritto nella sezione [3.1.2](#sec:methods). Durante l'esecuzione degli algoritmi vengono
stampati a schermo di volta in volta i risultati, come esemplificato
dall'immagine [2](#fig:cli-2).

![Output su terminale dopo l'esecuzione di `spa1.mtx` con tolleranza
0.0001.](images/output-cli.png){#fig:cli-2}

# Analisi dei risultati

Il programma sopra descritto ha ricevuto quindi come input le matrici di
test `spa1.mtx`, `spa2.mtx`, `vem1.mtx` e `vem2.mtx`. Per analizzare i
risultati dell'implementazione dei metodi è stato necessario prima di
tutto ottenere una panoramica delle proprietà di ciascuna matrice, come
si può osservare nella tabella
[1](#table:mat-stats).

| \textbf{Matrix} | \textbf{Size} | \textbf{Density} |
|-----------------|---------------|------------------|
| spa1.mtx        | 1000          | 0.182264         |
| spa2.mtx        | 3000          | 0.181304         |
| vem1.mtx        | 1681          | 0.004736         |
| vem2.mtx        | 2601          | 0.003137         |

L'esecuzione degli algoritmi implementati ha poi prodotto i risultati
rappresentati nelle tabelle [2](#table:spa1-stats),
[3](#table:spa2-stats),
[4](#table:vem1-stats) e
[5](#table:vem2-stats) divise per matrici.

Per un confronto più immediato si osservino i grafici a barre in figura
[3](#fig:tie-b), dove
vengono messi a confronto i metodi con il numero di iterazioni, il tempo
impiegato e l'errore relativo, divisi per tolleranza. Si noti che in
tutti i grafici a barre qui presentati la scala dell'asse y è
logaritmica.

![Grafici a barre del tempo di esecuzione, iterazioni ed errore relativo
di ciascun
metodo.](images/time-iterations-error_barplots.png)

Banalmente si può vedere che diminuendo la tolleranza cresce
esponenzialmente il tempo di esecuzione, il numero di iterazioni e
diminuisce esponenzialmente l'errore relativo. Più interessante è il
confronto tra i vari metodi e il tempo: Gauss-Seidel sembra essere il
metodo che impiega più tempo, mentre il gradiente coniugato è il più
veloce. Una possibile spiegazione della lentezza di Gauss-Seidel è
l'utilizzo dell'algoritmo della sostituzione in avanti non ottimizzata
per matrici sparse, che può aumentare significativamente il tempo di
esecuzione. Relativamente al numero di iterazioni è però il gradiente il
metodo che vede più iterazioni, di contro il gradiente coniugato
converge con un minimo numero di iterazioni.

Nonostante questa prima analisi dà già un'idea delle performance degli
algoritmi, osservando i dati nelle tabelle si può intuire una
correlazione tra la dimensione e la densità della matrice con il tempo e
il numero di iterazioni a seconda dei vari metodi impiegati. Confermiamo
questi sospetti osservando le matrici di correlazione in figura
[4](#fig:cormat).

![Matrici di correlazione dei risultati di ciascun
metodo.](images/heatmaps.png)

#### Jacobi

Il metodo di Jacobi vede una forte correlazione negativa tra densità e
iterazioni e una modesta correlazione sempre negativa tra densità e
tempo. La dimensione risulta invece pressoché ininfluente. Jacobi è
quindi adatto a matrici più dense di grandi dimensioni.

#### Gauss-Seidel

Gauss-Seidel si comporta similmente a Jacobi con una forte correlazione
negativa tra densità e iterazioni e tra densità e tempo. Anche
Gauss-Seidel è dunque adatto a matrici dense molto grandi.

#### Gradiente

Il metodo del gradiente invece trova una modesta correlazione positiva
tra densità e iterazioni e tra densità e tempo. La dimensione è invece
influente sul tempo con una correlazione positiva. Questo metodo è
quindi destinato alla risoluzione di matrici poco dense ma non di grandi
dimensioni.

#### Gradiente coniugato

Il gradiente coniugato accentua la tendenza del gradiente con una
correlazione positiva tra densità e iterazioni, tra densità e tempo e
tra dimensione e tempo. Anche il gradiente coniugato è pertanto adatto a
matrici poco dense e non di grandi dimensioni.

Le osservazioni fatte qui sopra vengono mostrate intuitivamente dai
grafici a barre nelle figure [5](#fig:di-b) e [6](#fig:dt-b).

![Grafici a barre del confronto tra densità e iterazioni per ciascun
metodo.](images/density-iterations_barplots.png)

![Grafici a barre del confronto tra densità e tempo per ciascun
metodo.](images/density-time_barplots.png)

| \textbf{Method}    | \textbf{Tolerance} | \textbf{Time (sec)} | \textbf{Iterations} | \textbf{Relative error} |
|--------------------|--------------------|---------------------|---------------------|-------------------------|
| Jacobi             | 0.0001             | 0.6279              | 115                 | 0.0017712811483049      |
| Gauss-Seidel       | 0.0001             | 1.2264              | 9                   | 0.018205942995187       |
| Gradient           | 0.0001             | 0.0671              | 143                 | 0.0345747007730108      |
| Conjugate Gradient | 0.0001             | 0.0421              | 49                  | 0.0207897600099575      |
| Jacobi             | 1e-06              | 0.8952              | 181                 | 1.797929543333586e-05   |
| Gauss-Seidel       | 1e-06              | 2.276               | 17                  | 0.000129969395857       |
| Gradient           | 1e-06              | 1.2214              | 3577                | 0.0009680457310584      |
| Conjugate Gradient | 1e-06              | 0.0955              | 134                 | 2.5529092778431404e-05  |
| Jacobi             | 1e-08              | 1.1803              | 247                 | 1.82497887635207e-07    |
| Gauss-Seidel       | 1e-08              | 3.2228              | 24                  | 1.7097328987836514e-06  |
| Gradient           | 1e-08              | 3.3928              | 8233                | 9.81636374476254e-06    |
| Conjugate Gradient | 1e-08              | 0.1219              | 177                 | 1.319840727794353e-07   |
| Jacobi             | 1e-10              | 1.5406              | 313                 | 1.8524371371887443e-09  |
| Gauss-Seidel       | 1e-10              | 4.1819              | 31                  | 2.248087839915377e-08   |
| Gradient           | 1e-10              | 4.7005              | 12919               | 9.820388418677605e-08   |
| Conjugate Gradient | 1e-10              | 0.1411              | 200                 | 1.207716797068718e-09   |


| \textbf{Method}    | \textbf{Tolerance} | \textbf{Time (sec)} | \textbf{Iterations} | \textbf{Relative error} |
|--------------------|--------------------|---------------------|---------------------|-------------------------|
| Jacobi             | 0.0001             | 1.0455              | 36                  | 0.0017662465191133      |
| Gauss-Seidel       | 0.0001             | 10.0109             | 5                   | 0.002598895587454       |
| Gradient           | 0.0001             | 0.696               | 161                 | 0.0181296451171125      |
| Conjugate Gradient | 0.0001             | 0.3088              | 42                  | 0.009821128457265       |
| Jacobi             | 1e-06              | 1.5683              | 57                  | 1.666756136718762e-05   |
| Gauss-Seidel       | 1e-06              | 16.0884             | 8                   | 5.1416412595227566e-05  |
| Gradient           | 1e-06              | 9.2137              | 1949                | 0.0006694229253797      |
| Conjugate Gradient | 1e-06              | 1.0808              | 122                 | 0.000119798461786       |
| Jacobi             | 1e-08              | 2.1628              | 78                  | 1.572869895211941e-07   |
| Gauss-Seidel       | 1e-08              | 21.7616             | 12                  | 2.794322031352431e-07   |
| Gradient           | 1e-08              | 20.5314             | 5087                | 6.865240127831838e-06   |
| Conjugate Gradient | 1e-08              | 1.3391              | 196                 | 5.586660595210111e-07   |
| Jacobi             | 1e-10              | 2.6861              | 99                  | 1.484271703014438e-09   |
| Gauss-Seidel       | 1e-10              | 27.3729             | 15                  | 5.570740916587933e-09   |
| Gradient           | 1e-10              | 35.2379             | 8285                | 6.937814826519222e-08   |
| Conjugate Gradient | 1e-10              | 1.673               | 240                 | 5.3242305205799435e-09  |


| \textbf{Method}    | \textbf{Tolerance} | \textbf{Time (sec)} | \textbf{Iterations} | \textbf{Relative error} |
|--------------------|--------------------|---------------------|---------------------|-------------------------|
| Jacobi             | 0.0001             | 5.878               | 1314                | 0.0035403807574038      |
| Gauss-Seidel       | 0.0001             | 74.7871             | 659                 | 0.0035069725970329      |
| Gradient           | 0.0001             | 0.0613              | 890                 | 0.0027045724093641      |
| Conjugate Gradient | 0.0001             | 0.0031              | 38                  | 4.082793158692197e-05   |
| Jacobi             | 1e-06              | 15.5031             | 2433                | 3.5400733429561766e-05  |
| Gauss-Seidel       | 1e-06              | 134.0643            | 1218                | 3.5266968493658966e-05  |
| Gradient           | 1e-06              | 0.0856              | 1612                | 2.7133391834267265e-05  |
| Conjugate Gradient | 1e-06              | 0.0036              | 45                  | 3.732339702219301e-07   |
| Jacobi             | 1e-08              | 16.8194             | 3552                | 3.539765956741338e-07   |
| Gauss-Seidel       | 1e-08              | 194.8304            | 1778                | 3.517456996804461e-07   |
| Gradient           | 1e-08              | 0.1472              | 2336                | 2.69533741911619e-07    |
| Conjugate Gradient | 1e-08              | 0.0044              | 53                  | 2.831873439816869e-09   |
| Jacobi             | 1e-10              | 21.6583             | 4671                | 3.5394587450468252e-09  |
| Gauss-Seidel       | 1e-10              | 255.6368            | 2338                | 3.508242322543687e-09   |
| Gradient           | 1e-10              | 0.1664              | 3058                | 2.713166874216703e-09   |
| Conjugate Gradient | 1e-10              | 0.0048              | 59                  | 2.1917517962458478e-11  |


| \textbf{Method}    | \textbf{Tolerance} | \textbf{Time (sec)} | \textbf{Iterations} | \textbf{Relative error} |
|--------------------|--------------------|---------------------|---------------------|-------------------------|
| Jacobi             | 0.0001             | 21.1842             | 1927                | 0.0049684614061959      |
| Gauss-Seidel       | 0.0001             | 166.5856            | 965                 | 0.004951189291546       |
| Gradient           | 0.0001             | 0.1007              | 1308                | 0.0038119295293938      |
| Conjugate Gradient | 0.0001             | 0.005               | 47                  | 5.729017222235908e-05   |
| Jacobi             | 1e-06              | 41.432              | 3676                | 4.9670344320611774e-05  |
| Gauss-Seidel       | 1e-06              | 320.5974            | 1840                | 4.9417612682488e-05     |
| Gradient           | 1e-06              | 0.1817              | 2438                | 3.791418978880894e-05   |
| Conjugate Gradient | 1e-06              | 0.0069              | 56                  | 4.742996283356508e-07   |
| Jacobi             | 1e-08              | 60.7174             | 5425                | 4.965607902481058e-07   |
| Gauss-Seidel       | 1e-08              | 467.1627            | 2714                | 4.958369689706862e-07   |
| Gradient           | 1e-08              | 0.2431              | 3566                | 3.809850623663889e-07   |
| Conjugate Gradient | 1e-08              | 0.0071              | 66                  | 4.299983499500873e-09   |
| Jacobi             | 1e-10              | 80.6605             | 7174                | 4.964185220415289e-09   |
| Gauss-Seidel       | 1e-10              | 633.3592            | 3589                | 4.948912856957899e-09   |
| Gradient           | 1e-10              | 0.3212              | 4696                | 3.798767935221748e-09   |
| Conjugate Gradient | 1e-10              | 0.009               | 74                  | 2.2476276371368195e-11  |


# Conclusioni

Gli obiettivi implementativi sono stati raggiunti: l'unica nota dolente
la ritroviamo nei tempi di esecuzione di Gauss-Seidel con matrici poco
dense, probabilmente a causa dell'implementazione della sostituzione in
avanti nell'update del metodo.

L'interfaccia da linea di comando dell'eseguibile consente una buona
facilità di utilizzo e la produzione di grafici risulta molto efficace
per un'analisi post esecuzione.

In conclusione, i metodi implementati in questo progetto vedono
preferire l'utilizzo di Jacobi e Gauss-Seidel per matrici poco dense ma
molto grandi, mentre gradiente e gradiente coniugato per matrici più
dense ma meno grandi.
