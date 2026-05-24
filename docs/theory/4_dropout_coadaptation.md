## Dropout (regularisation technique to prevent overfitting) / Co-adaptation is the problem that Dropout fixes:

- dropout randomly turns some neurons off, so that during the training, we can reduce the co-adaptation
- Co-adaptation happens when a neuron relies on another neuron 
- Relying may happen due to neuron A firing all the time, and neuron B copies it
- An analogy would be we have a table mate who is smart. So every test, we can just copy from him.
- But what dropout does it that it increases the chance that he might not be there, meaning we need to actually think on our own and rely on ourselves for an output.
- Hence, decreasing overfitting (everyone in class getting the same model answer), and decreasing co-adaptation where all of us rely less on one another and have our own way of thinking (adaptability).