# Using Food for Data Science Instruction
This is repository consist of all the code and report on my work for Capstone Project done at Rochester Institute of Technology.

# Abstract

Using difficult topics when teaching machine learning can induce anxiety in students and cause them to mentally distant from the subject matter resulting in a lack of teaching effectiveness.
Topics such as cancer, pandemics, mortality can distract the listener from the key points. Using familiar examples such as food, the students can learn the methodology without becoming distracted because it is more relatable.  This paper demonstrates methods of teaching machine learning using examples drawn from food and cooking.

# Background 

Due to the current pandemic, there is a lot to be learned from the spread of viruses such as modeling viral spreading, exponential growth, and geometric decay. However, obviously, many people find it disturbing to talk about the virus because of the mortality rate associated with it. Instead of talking about a virus such as COVID-19, we can talk about food trends or fads. Fads can behave like a virus. They have a quick rise and steep decline in a short amount of time. Here are two or three food fads that demonstrate this. The following examples are based on the presumption that when someone searches on a topic that they are interested in it.

An example of food fad is Dalgona coffee induced by the isolation during COVID-19. As per Youtube searches on Google Trends~\cite{google}, \emph{“How to make Dalgona coffee”} has become the most searched type of coffee recipe worldwide. As shown in Figure below, searches worldwide for Dalgona coffee recipe peaked in mid-April and declined by the end of April.
![Interest in Dalgona Coffee over time](/images/dalgona.png)

Another example of a food fad was the increase in popularity of Kale. It was coined as “superfood” and is found now as chips or in salads. From Figure below, it can be seen that Kale became very popular in the United States in mid-2014 when young-adults where being drawn to a more health-conscious diet. It is now slowly declining, and its popularity is reduced by half since 2014.
![Interest in Kale over time](/images/kale.png)

A traditional Japanese tea, Matcha, has been gaining a lot of popularity lately as well. Matcha has become attractive to millennials because of green color of the drink being aesthetically pleasing as well the health benefits such as weight loss. In Figure below, it can be seen how the popularity of Matcha has been increasing over the last few years, even though it has been around since the 7th Century.

![Interest in matcha over time](/images/matcha.png)

# Survey

To validate our belief, we conducted a survey in which we asked the students various questions about their topic preferences when learning Data Science. They were questioned whether they would prefer learning about food topics or cancer, food or pandemics, and food or climate change. 

Using Weka, K means was performed on data collected, and on analysis, three clusters were found. From Table below, we can see the information generated by Weka. If the value in the table is zero or closer to zero, it indicates that the preference is not food. If the value is closer to 1 or 1, the preference is food. It shows that there are 43% of students in the class who are comfortable talking about grim topics as data science can be used to save lives. Also, it can be seen that 26% of the students do not want to talk about morbid topics and it can cause communication and learning blocks in these students. The rest of the students were neural in their preference.

Attribute | Cluster 0 | Cluster 1 | Cluster 2 
--------- | --------- |-----------| ---------
Food over Biological Processes | 0 | 0.9 | 1
Food over Pandemics or Epidemics | 0.03 | 1 | 0

The above information tells us that there is a mixture model of students in the class. While some have a morbid curiosity, others prefer not to talk about grim topics. Also, from Figure below, we can observe that most students like food. In order to include the 26% of the students and promote learning without any communication block, data science should be taught with topics such as food. Thus, proving our original hypothesis. 

![Food Preference](/images/pie.png)

# Example Machine Learning Algorithms using Food

## Background on Gradient Descent
Gradient Descent is an iterative optimization algorithm that takes more significant steps where the amount of gradient is higher or moves in the direction of steepest descent. It is used to find the parameters of a model in machine learning algorithms such as Logistic Regression, SVM, etc. An analogy for gradient descent would be a ball rolling down a hill. The ball would initially roll down faster till it slows down when it is about to reach a dip. The goal of gradient descent is finding this *dip*, known as the local minimum. The first step is to initialize a starting point. Then steps are taken in path of the greatest descent. The rate at which these *steps* are taken is called the learning rate. 
The two parameters required are -
- Learning Rate: It is used to control the change in each coefficient. It should not be too high as it is possible to jump over the bottom, and if it too low, then the algorithm takes a longer time to run.
- Epochs: The number of iterations that need to be performed. This amount can vary from dataset to dataset.

The Gradient Descent calculated also depends on the cost function. The cost function tells how *good* the model is. The two parameters of the cost function are weight and bias. The slope of the cost function informs on how to update the parameters.

## Grid Search
Similar to Gradient Descent, we can use as Grid Search as well to find parameters.  In a grid search, a rough set of all parameters is tried. Then, once a global best fit is found, the parameters in that region are iteratively changed to improve the fit to the local model, and the precision of the model.
Figure below a SIR (Susceptible-Infected-and-Recovered) model is used to predict Dangola coffee. 

![Grid Search](/images/coffee.png)

Before getting into Logistic Regression, it is vital to understand the concept of Odds and Odds Ratio. Odds are the ratio of success to the ratio of failure. The range of odds can be in the range of 0 to infinity. 
If we take a natural logarithm of such numbers, for a number $x\geq0$, $\log(x)$ is in the range of [-infinity, infinity].  Odds ratios, as the name suggests, is a ratio of odds. The odds ratio can vary between 0 to positive infinity, $\log (Odds Ratio)$ can vary between [-infinity,infinity]. Specifically, when the odds ratio lies between [0,1], log(Odds Ratio) is negative.

## Logistic Regression

The term logistic regression is confusing as it is neither logical nor a typical regression. Logistic regression is a statistical classification method for predicting binary cases, i.e., the target class is either 1 or 0. To explain it better, we are using data on popcorn to predict two scenarios: if the popcorn is cooked or not, and if the popcorn is burnt or not. If the popcorn is cooked, then the target class is 1, and if uncooked, it is 0. Since the values predicted by logistic regression are binary, they are mainly used for classification purposes. However, the output of logistic regression, is the probability of the prediction i.e values are in the range of 0 to 1. This is used to find how similar the predicted value is to the expected value. It is assumed that all values above 0.5 belong to Class 1, and all values below 0.5 belong to Class 0. Here, we are trying to classify for a given time in seconds, what is the probability that the popcorn is cooked. Suppose value for x1 value is probable to 0.8, then there is an 80% probability that it is cooked. 

As the values that need to be predicted by logistic regression should be in the range of 0 to 1, we use sigmoid function. Here *w* is the weight, and *b* is a bias.

\[
\ h_{w,b}\left(x\right)=\frac{1}{1+e^{-{(w}^\top.x+b)}}
\]

\begin{figure}[!t]
\centering
\includegraphics[width=3.0in]{sig}
\caption{An example of Sigmoid function. The vertical red lines show the error between the function and the data.}
\label{fig:sig}
\end{figure}

The sigmoid function can be used to create a sigmoidal curve, as shown in Figure~\ref{fig:sig}. By examining the curve in Figure~\ref{fig:sig}, we can see that the error would be decreased if the sigmoid function was shifted further to the right. The problem with machine learning is to figure out how to get a computer to examine this situation and then have the algorithm decide to shift the curve to the right. The secret is to have the algorithm consider the amount of error and the slope of the sigmoid function at which it occurs. In this case, since the error occurs where the sigmoid function is going up, the curve should be shifted to the right such that error lines intersect sigmoid at a lower value. 

The cost function of logistic regression is $ Cost\left(h_\theta\left(x\right),y\right)=-y\ \ log\left(h_\theta\left(x\right)\right)\ -(1-y) log\left({1-h}_\theta\left(x\right)\right)\ $~\cite{ganesh_2019}

It is a convex function. The right and left part of the cost function join together to form a bowl shape, like the one in Figure~\ref{fig:gradient}.
\[
   Cost(h_\theta\left(x\right),y)=\begin{cases}
    -\log(h_\theta x), & \text{if y=1}.\\
    -\log(1-h_\theta x), & \text{if y = 0}.
  \end{cases}~\cite{ganesh_2019}
\]

The intention is to minimize the cost function. In order to achieve this, we repeatedly update the weights and bias using gradient descent until the global minimum is reached. In other words, to fix the error in Figure~\ref{fig:sig}, we can use gradient descent. 

Every iteration of logistic regression is used to update the coefficients.
In each iteration, the error is found between the predicted value and expected value. Based on this error and the learning rate, the coefficients are updated. Eventually, the error lowers to an insignificant amount.
Once the lowest cost is reached, the final coefficients can be used to make predictions. The above algorithm was written for the popcorn data, and it generated an accuracy of 0.83.

Figure~\ref{fig:alog} shows the result of logistic regression on popcorn data. The blue sigmoidal curve is for the cooked data, and the orange is for burnt data.  Two logistic regression curves can be used to give a clever function. This is what artificial neurons do. A delta function can be drawn between the two curves. Using the delta function, any curve can be generated. This is why artificial neural networks can be used as a universal approximator.

![Logistic Regression](/images/log.png)


## Support Vector Machines
Support Vector Machines are a supervised machine learning algorithm that is used to make binary classification. The SVM algorithm tries to find a decision boundary using support vectors that separates the data such that on the side lies one category, and on the other side lies the other category. These decision boundaries are called Hyperplanes. There can  be  multiple  hyperplanes  that  separate  the  data,  but  the ideal one should satisfy the following conditions:

- It creates divide between the classes with a maximum margin.
- item Its equation generates a value greater than 1.

Hyperplanes ensures that future data is classified with more confidence. The data points that are closest to the hyperplane are support vectors. They can change the orientation or maximize the margin between the hyperplane and the data points. 
The hyperplane can be found by finding ideal parameters, \emph{weight} and \emph{bias} by minimizing the cost function. The cost function~\cite{zha_2019} is 

\[
\ J(w) = \frac{1}{2}||w||^2 + C\left[\frac{1}{N} \sum_{i}^{n} max(0,1 - y_i*(w.x_i + b))\right]
\]

We can minimize the cost function by minimizing $||w||^2$, which minimizes the margin, or we can minimize the sum of hinge loss~\cite{zha_2019} $max(0,1 – y_i*(w.x_i + b)$ . We can minimize either of the two by using gradient descent. 

\begin{figure}[!t]
\centering
\includegraphics[width=3.0in]{svm}
\caption{Support Vector Machines used to classify if given recipe is a Muffin or Cupcake. The bold line represents the Hyperplane.}
\label{fig:svm}
\end{figure}

Consider the graph in Figure~\ref{fig:svm}; here SVM is used to determine if a given recipe is that of a cupcake or a muffin. The features used are the amount of Sugar and Flour in a recipe. The prediction is labeled as 0 and 1 for Muffin and Cupcake respectively. The accuracy of the model resulted was 1.0. And the ideal hyperplane is the one shown in Figure~\ref{fig:svm}.

![Support Vector Machines](/images/svm.png)

## K means

K means is a clustering algorithm; a clustering algorithm divides the data into subgroups where a point in a subgroup is similar to other points in that subgroup. This similarity is found using distance metrics such as Euclidean distance. Unlike the supervised methods mentioned above, clustering is unsupervised.
K means is an iterative method that divides the data into \emph{k} pre-defined cluster/subgroups, where each point in data belongs to exactly one cluster. Each cluster has a center called centroid: it is the average of all the points in the cluster. Once centroid is initialized, points are assigned to each cluster such that the sum of squared distance between all points in that cluster and center is at a minimum. The center is re-calculated, and this process is repeated iteratively till the center of the clusters stops changing. The algorithm in steps:
- Specify *k*.
- Initialize the centroids and assign points to the closest cluster.
- Keep iterating till the center of the cluster stops changing, or a seed limit is reached.
- Compute the sum of squared distance between the center and all data points.
- Assign points to the closest center.
- Compute the center.

It is essential to select the correct *k* to get homogeneous clusters. A *knee* graph (some call it an elbow curve) can be plotted to find the ideal *k*. The idea is to find the sum of squared error for a range of k values and then plot the sum of squared error for each value of k. The ideal *k* is point right after which there is a steady, almost linear decrease in the graph. Observe, Figure~\ref{fig:knee}, notice that six is ideal *k*. Here, K means was used to do market basket analysis. The data consist of different shoppers that buy items from a store. The goal is to find the food groups that the shoppers belong to, such as keto, vegan, gluten-free, etc. Based on Figure~\ref{fig:knee}, we see six groups emerge. 

![Knee Graph](/images/knee.png)

## Agglomeration
Agglomeration or agglomerative clustering is a bottom-up approach to hierarchical clustering. Here, each point in the data is considered as an individual cluster and merged until there is only one cluster. One of the key decisions is to choose a linkage. It is required to determine distance between the points. Different types of linkage are the ward’s method, complete linkage, single linkage, and average linkage. Another decision is to select the metric to calculate the linkage. It is usually a Euclidean distance. Agglomeration algorithm is as follows:
- Compute the distance between each data point.
- Each data point is its own cluster at the start.
- Merge two closest clusters and re-compute the distance.
- Repeat the previous step till all the clusters are merged into one.

Dendrograms help understand agglomeration in a much better and more visual way. It shows in a static way how the aggregations are performed or how each cluster is joined together until there is only cluster left. The height of the dendrogram indicates the order in which the clusters were joined, and it can also be used to determine how far apart each cluster is. The more the height before joining, the further apart they are. Dendrograms can also be used to find the number of clusters by cutting it by drawing a horizontal line across it.  It is important to know where to cut the line. The line must be cut where the difference is most significant, or clusters are far apart—this line when drawn across dendrogram cut lines in the dendrogram. The number of lines it cuts is the number of clusters. 

![Dendogram](/images/agg.png)

When deciding between K means or Hierarchical clustering, usually, Hierarchical Clustering is preferable. It is because Hierarchical clustering has fewer hidden assumptions about the distribution of data.  
With K means clustering, the desired number of clusters should be known. Also, k-means will often give unintuitive results if the data is not well-separated into sphere-like clusters, *k* picked is not well-suited to the shape of the data or initial values for the cluster centroids are weird.

The only requirement (which k-means also shares) of hierarchical clustering is that a distance has to be calculated between each pair of data points. Hierarchical clustering typically joins nearby points into a cluster, and then successively adds nearby points to the nearest group. A dendrogram can be used to decide how many clusters your data has, by cutting the dendrogram at different heights. Of course, it has to be pre-decided on how many clusters one wants.

## Artifical Neural networks
Artificial Neural Networks is a computational model that has a similar structure of the neural network of a human brain. Like in the human brain, there is a neuron called a node. There can be multiple layers in the network. The node in one layer is connected to all the nodes in the subsequent layer. The signal in this network flows from left to right. The node, when given an input, implements a function and forwards the output to the next layer. Each input is given a weight. This weight depends on the significance of the input. The resulting output is generated after all the nodes complete the above process. There are various kinds of neural networks, but the most commonly used ones
are:
- Feedforward
- Recurrent


Artificial Neural Network is used to find the classification between muffins and cupcakes. It similarly classifies the data as Support Vector Machines but takes slightly longer. This shows that we should not use ANN unless it is required, and it better to go with the more straightforward option of SVM (Occam’s Razor). 

# Conclusion
We investigated the hypothesis that students would not want to discuss grim topics. In order to do that, we conducted an experiment where the students answered questions based on their topic preferences. K means was performed on the data collected from the experiment. We identified that there were three clusters of students with different preferences. 
As expected the first cluster did not want to talk about sickness, mortality, extinction, or anything grim. However, this cluster comprised only 26\% of the students.
A second cluster of students either had morbid curiosity, or who want to use data science to save lives. This consisted of 46\% of students. The remaining cluster of student were neutral.

In this paper, we have shown five examples demonstrating the use of food to teach machine learning. These include commonly used classification and clustering algorithms. The classification methods include: Logistic Regression, Support Vector Machines,Artificial Neural Network. The clustering techniques demonstrated are K means, Agglomeration.

We used logistic regression to classify how long it takes for a bag of popcorn to get cooked. Support Vector Machines are used to classify a given recipe as a cupcake recipe or a muffin recipe. For comparison purposes, Artificial Neural Networks were used on the same cupcake recipe or muffin recipe task. This helps students compare and contrast two algorithms. For finding if a recipe is a muffin or cupcake, it is better to use support vector machines as they are easier to implement and generate the same classification as an Artificial Neural Network. Additionally, Support Vector Machines run much faster.

To demonstrate K means and Agglomeration, we used market basket analysis to identify subgroups of shoppers in a grocery store. This helps in recommending products to a particular subset of shoppers. This type of clustering algorithms can be used to create recommender algorithms that help in issuing coupons to shoppers.

Overall, we found a cluster of students who did not mind discussing unpleasant topics (43\%), and
we identified a cluster of around one quarter of students who prefer learning data science while avoiding unpleasant topics (26\%). 
In order to help the latter students understand machine learning better, we have given recommended approaches using pleasant topics.
These example given include the use of food, cooking, recipes, and shopping. 
Using these techniques, data science can be taught to the full spectrum of students while avoiding communication and learning blocks. At the start of this project, there was no indication that there would be a Coronovirus pandemic. The mortality of the pandemic further justifies the need to avoid communication blocks. 
