## Workshop 4: Bayes Nets

In this assignment, you will work with probabilistic models known as Bayesian networks to efficiently calculate the answer to probability questions concerning discrete random variables.


### Setup

1. Clone the project repository from Github

   ```
   git clone this_repo
   ```

Substitute your actual username where the angle brackets are.

2. Navigate to directory

3. Activate the environment 

    ```
    conda activate env
    ```
    
    In case you used a different environment name, to list of all environments you have on your machine you can run `conda env list`.

4. Run the following command in the command line to install and update the required packages

    ```
    pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
    pip install --upgrade -r requirements.txt
    ```


## Part 1 Bayesian network tutorial:

The problem below is designed to view how materials in the course could be used in natural sciences. You will solve a Bayesian
network question using similar principles from A3, where the variables are bolded. Paragraphs that serve as extra information to
provide context to the reader are in italics. The team understands that sometimes probability notation can b e confusing, so we
have tried to be as clear as possible to avoid any confusion.
The 􀃘eld of molecular biology has a concept called the "central dogma of molecular biology". This dogma states that the 􀃙ow of genetic
information goes from DNA to RNA, and then the information from RNA gets translated into proteins. In eukaryotes (e.g. mammals, birds,
reptiles), DNA is protected in the nucleus of the cells, and an RNA copy has to be made in order to generate proteins. Proteins can be used
for a variety of biological applications. They can serve as a receptors (e.g. receptors in the cells in our noses to detect different smells), as
enzymes (e.g. lactose-intolerant people can add lactase in milk to break down lactose, allowing them to consume milk), as hormones (e.g.
insulin for diabetic individuals), and many other cases. For example, the mRNA COVID vaccines are used to generate the spike proteins in
the coronaviruses (which when assembled in a full viral capside, it’s used to penetrate the host cells, causing infection). For simplicity, we
will use the term "protein" as a generic term to also include "peptides".

To start, design a basic probabilistic model for the following system:

For this problem, we will be modeling the synthesis of proteins and their stability inside cells. 
To keep the previous paragraphsimple, DNA is used to create RNA, and RNA is used to create proteins. 
In order to create RNA, we need to activate a gene (G).This essentially means turning a gene ON from its OFF state. 
This will lead to the creation of RNA, and then RNA can be used tocreate proteins. 
Proteins, however, can often have stochastic behaviors, especially due to is polymeric structure. 
In order for aprotein to have its expected function, it needs to have a correct folding (F), which in turn will lead to have some stability (S).
Unfortunately, mutations (M) can occur at the DNA or at the RNA. Sometimes mutations can be negligent, but sometimes, amutation (M) can lead to an incorrect folding (i.e. not correct folding (F)). 
After a protein has been synthesized, it will begin itsfunction, but proteins have an "expiration date". These expiration dates often are dictated by other factors inside cells (e.g.enzymes). If a protein is not correctly, folded, it will not be stable. Also, after some time, these other factors can perform a hydrolysis (H) on the protein and destabilize the protein, ceasing its function. 
Finally, ligands (L) can be introduced to cells thatcan infl uence the activation of genes (G) either directly or indirectly. For example, certain drugs can be used as ligands to specifi ccell receptors that induce a cascade of events that ultimately activates a gene.

### 1a: Casting the net

This means we have 6 variables:
- L: ligand
- G: gene activation
- M: mutation
- F: correct folding
- H: hydrolysis
- S: stability

  
There are therefore several probability tables. Below are their defi nitions, where we assign boolean values to each of thevariables, and the tables are listed below the defi nitions
- p(L = 1): probability of a ligand being present
- p(M = 1): probability of a mutation occurring
- p(H = 1): probability of a protein being hydrolyzed
- p(G = 1|L): probability of a gene being activated given the presence of a ligand
- p(F = 1|M, G): probability of the protein correctly folding given the presence of a mutation and the gene being activated
- p(S = 1|H, F): probability of the protein being stable, given the occurrence of a hydrolysis and the protein being correctlyfolded

<table>
<tr><th>  </th><th> </th></tr>
<tr><td>

L | P(L)
--- | --- 
1 | 0.7
0 | 0.3

</td><td><td>


M | P(M)
--- | --- 
1 | 0.3
0 | 0.7

</td></tr> </table>

<table>
<tr><th>  </th><th> </th></tr>
<tr><td>

H | P(H)
--- | --- 
1 | 0.4
0 | 0.6

</td><td><td>


L | P(G=1 \| L) | P(G=0 \| L) 
--- | --- | --- 
1 | 0.9 | 0.1
0 | 0.2 | 0.8

</td></tr> </table>

M | G | P(F=1 \| M, G) | P(F=1 \| M, G) 
--- | --- | --- | --- 
1 | 1 | 0.9 | 0.1
0 | 0 | 0.2 | 0.8
1 | 0 | 0.9 | 0.1
0 | 0 | 0.2 | 0.8


Use the description of the model above to design a Bayesian network for this model. The `pgmpy` package is used to represent nodes and conditional probability arcs connecting nodes. Don't worry about the probabilities for now. Use the functions below to create the net. You will write your code in `submission.py`. 

Fill in the function `make_security_system_net()`

The following commands will create a BayesNet instance add node with name "node_name":

    BayesNet = BayesianNetwork()
    BayesNet.add_node("node_name")

You will use `BayesNet.add_edge()` to connect nodes. For example, to connect the parent and child nodes that you've already made (i.e. assuming that parent affects the child probability):

Use function `BayesNet.add_edge(<parent node name>,<child node name>)`.  For example:
    
    BayesNet.add_edge("parent","child")

After you have implemented `make_security_system_net()`, you can run the following test in the command line to make sure your network is set up correctly.

```
python probability_tests.py ProbabilityTests.test_network_setup
```

### 1b: Setting the probabilities

_[15 points]_

Now set the conditional probabilities for the necessary variables on the network you just built.

Fill in the function `set_probability()`

Using `pgmpy`'s `factors.discrete.TabularCPD` class: if you wanted to set the distribution for node 'A' with two possible values, where P(A) to 70% true, 30% false, you would invoke the following commands:

    cpd_a = TabularCPD('A', 2, values=[[0.3], [0.7]])

**NOTE: Use index 0 to represent FALSE and index 1 to represent TRUE, or you may run into testing issues.**


If you wanted to set the distribution for P(G|A) to be

|  A  |P(G=true given A)|
| ------ | ----- |
|  T   | 0.75|
|  F   | 0.85| 

you would invoke:

    cpd_ga = TabularCPD('G', 2, values=[[0.15, 0.25], \
                        [ 0.85, 0.75]], evidence=['A'], evidence_card=[2])

**Reference** for the function: https://pgmpy.org/_modules/pgmpy/factors/discrete/CPD.html

Modeling a three-variable relationship is a bit trickier. If you wanted to set the following distribution for P(T|A,G) to be

| A   |  G  |P(T=true given A and G)|
| --- | --- |:----:|
|T|T|0.15|
|T|F|0.6|
|F|T|0.2|
|F|F|0.1|

you would invoke

    cpd_tag = TabularCPD('T', 2, values=[[0.9, 0.8, 0.4, 0.85], \
                        [0.1, 0.2, 0.6, 0.15]], evidence=['A', 'G'], evidence_card=[2, 2])

The key is to remember that first entry represents the probability for P(T==False), and second entry represents P(T==true).

Add Tabular conditional probability distributions to the bayesian model instance by using following command.

    bayes_net.add_cpds(cpd_a, cpd_ga, cpd_tag)


You can check your probability distributions in the command line with

```
python probability_tests.py ProbabilityTests.test_probability_setup
```

### 1c: Probability calculations : Perform inference

_[10 points]_

To finish up, you're going to perform inference on the network to calculate the following probabilities:

>- What is the marginal probability that the “Double-0” files get compromised? 
>- You just received an update that the British Elite Forces have successfully secured and shut down Contra, making it unavailable for Spectre. Now, what is the conditional probability that the “Double-0” files get compromised?
>- Despite shutting down Contra, MI6 still believes that an attack is imminent. Thus, Bond is reassigned full-time to protect M. Given this new update and Contra still shut down, what is the conditional probability that the “Double-0” files get compromised?

You'll fill out the "get_prob" functions to calculate the probabilities:
- `get_marginal_double0()`
- `get_conditional_double0_given_no_contra()`
- `get_conditional_double0_given_no_contra_and_bond_guarding()`

Here's an example of how to do inference for the marginal probability of the "A" node being True (assuming `bayes_net` is your network):

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['A'], joint=False)
    prob = marginal_prob['A'].values
  
To compute the conditional probability, set the evidence variables before computing the marginal as seen below (here we're computing P('A' = false | 'B' = true, 'C' = False)):


    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['A'],evidence={'B':1,'C':0}, joint=False)
    prob = conditional_prob['A'].values
    
__NOTE__: `marginal_prob` and `conditional_prob` return two probabilities corresponding to `[False, True]` case. You must index into the correct position in `prob` to obtain the particular probability value you are looking for. 

If you need to sanity-check to make sure you're doing inference correctly, you can run inference on one of the probabilities that we gave you in 1a. For instance, running inference on P(M=false) should return 0.20 (i.e. 20%). However, due to imprecision in some machines it could appear as 0.199xx. You can also calculate the answers by hand to double-check.
