# Maximum Likelihood Estimation with Principal Component Analysis for Multiclass Classification

Firstly, 𝐷𝑐 and 𝑇𝑐 sets were created by splitting each class of the given **private** dataset to 75% and 25% and then they were merged to 𝐷𝑖 & 𝑇𝑖, while 𝑐 = 1,2,3,…8. In this project, dimension of the dataset was reduced with PCA,shape of the dataset was changed from (987,600) to (987,120). PCA applied separately on each feature type that consisting 1x100 independent feature vector.

## Procedure:

For each feature type with 100 dimensions, PCA was applied separately. Dimensions were reduced to 20 for each feature type. For multivariate gaussian distribution with unknown mean and covariance matrix, MLE method was used to estimate parameters in a fixed way. Since we know that for multivariate Gaussian case, the maximum likelihood estimate for the mean vector is the sample mean and the maximum likelihood estimate for the covariance matrix is the arithmetic average of the 𝑛 matrices (𝑥𝑘−𝜇̂)(𝑥𝑘−𝜇̂)𝑡.

<p align="center">
  <img src="https://i.ibb.co/BKRYkJ5/Screenshot-1.png">
</p>

After estimated parameters are calculated, we can compute the posterior:

<p align="center">
  <img src="https://i.ibb.co/Drdh88m/Screenshot-2.png">
</p>
To assign a sample vector 𝑥 to a class with minimum, we can use discriminant function with minimum error-rate classification,

<p align="center">
  <img src="https://i.ibb.co/DCMNBjz/Screenshot-3.png">
</p>
For each class, we can compare resulting pdfs’ multiplied by prior probabilities in natural logarithm. While 𝑔𝑖(𝑥⃗),

                                                𝑔𝑖(𝑥⃗) = 𝑙𝑛 𝑃(𝑥⃗ | 𝑤𝑖) + 𝑙𝑛 𝑃(𝑤𝑖)

After iterating over all classes’ resulting if, 𝑔𝑖(𝑥⃗) > 𝑔𝑗(𝑥⃗) then we assign x to the class 𝑤𝑖.

Precision and recall metrics are calculated as follows,

<p align="center">
  <img src="https://i.ibb.co/sb8FCcY/Screenshot-4.png">
</p>
Precision and recall results for 𝑇𝑖 set:

<p align="center">
  <img src="https://i.ibb.co/bFzT9mZ/Screenshot-5.png">
</p>

## Note: 
For overall comparison of algorithms on same **private dataset** please visit [Hierarchical Clustering with SVM](https://github.com/mburakbozbey/hierarchical-clustering-with-svm) repository.
