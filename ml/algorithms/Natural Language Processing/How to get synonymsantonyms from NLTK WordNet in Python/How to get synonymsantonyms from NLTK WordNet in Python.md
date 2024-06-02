# How to get synonyms/antonyms from NLTK WordNet in Python?
How to get synonyms/antonyms from NLTK WordNet in Python?
---------------------------------------------------------



WordNet is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations.  
**WordNet’s structure makes it a useful tool for computational linguistics and natural language processing.**

WordNet superficially resembles a thesaurus, in that it groups words together based on their meanings. However, there are some important distinctions.

*   First, WordNet interlinks not just word forms—strings of letters—but specific senses of words. As a result, words that are found in close proximity to one another in the network are semantically disambiguated.
*   Second, WordNet labels the semantic relations among words, whereas the groupings of words in a thesaurus does not follow any explicit pattern other than meaning similarity.

`from` `nltk.corpus` `import` `wordnet`

`syns` `=` `wordnet.synsets(``"program"``)`

`print``(syns[``0``].name())`

`print``(syns[``0``].lemmas()[``0``].name())`

`print``(syns[``0``].definition())`

`print``(syns[``0``].examples())`

**The output will look like:**  
plan.n.01  
plan  
a series of steps to be carried out or goals to be accomplished  
\[‘they drew up a six-step plan’, ‘they discussed plans for a new bond issue’\]

Next, how might we discern synonyms and antonyms to a word? The lemmas will be synonyms, and then you can use .antonyms to find the antonyms to the lemmas. As such, we can populate some lists like:

`import` `nltk`

`from` `nltk.corpus` `import` `wordnet`

`synonyms` `=` `[]`

`antonyms` `=` `[]`

`for` `syn` `in` `wordnet.synsets(``"good"``):`

    `for` `l` `in` `syn.lemmas():`

        `synonyms.append(l.name())`

        `if` `l.antonyms():`

            `antonyms.append(l.antonyms()[``0``].name())`

`print``(``set``(synonyms))`

`print``(``set``(antonyms))`

**The output will be two sets of synonyms and antonyms**  
{‘beneficial’, ‘just’, ‘upright’, ‘thoroughly’, ‘in\_force’, ‘well’, ‘skilful’, ‘skillful’, ‘sound’, ‘unspoiled’, ‘expert’, ‘proficient’, ‘in\_effect’, ‘honorable’, ‘adept’, ‘secure’, ‘commodity’, ‘estimable’, ‘soundly’, ‘right’, ‘respectable’, ‘good’, ‘serious’, ‘ripe’, ‘salutary’, ‘dear’, ‘practiced’, ‘goodness’, ‘safe’, ‘effective’, ‘unspoilt’, ‘dependable’, ‘undecomposed’, ‘honest’, ‘full’, ‘near’, ‘trade\_good’} {‘evil’, ‘evilness’, ‘bad’, ‘badness’, ‘ill’}

**Now , let’s compare the similarity index of any two words**

`import` `nltk`

`from` `nltk.corpus` `import` `wordnet`

`w1` `=` `wordnet.synset(``'run.v.01'``)`

`w2` `=` `wordnet.synset(``'sprint.v.01'``)`

`print``(w1.wup_similarity(w2))`

Output:  
0.857142857143

`w1` `=` `wordnet.synset(``'ship.n.01'``)`

`w2` `=` `wordnet.synset(``'boat.n.01'``)`

`print``(w1.wup_similarity(w2))`

Output:  
0.9090909090909091

 