# Neural-Dissected Decision Tree (NDT)
A novel surrogate model called Neural-Dissected Decision Tree (NDT) to inhibit the Rashomon effect of MLPs. Our paper is currently under review by IEEE Transactions on Computational Social Systems.

![The pipeline of the proposed Tree Explanation for Rashomon effect of MLPs. It contains three main procedures:Strategy Analyzing, Tree Building and Tree Evaluation.](./main.png)

## What is the Neural-Dissected Decision Tree?
Surrogate models are commonly adopted in explainable artificial intelligence (XAI). By Learning the relationships between the inputs and outputs of the black-box model, surrogate models such as decision trees can be used to inspect the intrinsic decision-making of the host model. However, different surrogate models or random initialization states may result in different explanations, which is called the Rashomon effect. For this problem, we propose a novel surrogate model called Neural-Dissected Decision Tree (NDT) to inhibit this effect of the deep-learning models.  


<p align="center">
<img src="./introduction.png" width=80%>
</p>
<p align="center" style="color:grey;font-size:90%">
Both decision trees are extracted from the multi-layer perceptron (MLPs). However, the decision-maker is
confused when confronted with two tree explanations that
are approximately equally accurate.
</p>

## Better Explanation
As Rtree is built in a strategy-level order manner which synchronized with the hidden-layers, users can better understand the decision-making of the deep-learning model through visualization.


<p align="center">
<img src="./decision_tree.png" width=45%>    <img src="./our_tree.png" width=45%>
</p>

<p align="center" style="color:grey;font-size:90%">
decision tree (left) vs Rtree (right)
</p>

## Usage
Please refer to the Find01_example.ipynb for details.
