# ProbabilisticRrobotics
Implementation of probabilistic robotics book

Author: Minwoo shin
Date: 2021-Aug-01

Problem01 @ 04/Aug/21
At first step, The algorithm find proper correspondences(with small noise)
But, in the second step, The correspondence variable algorithm(Line 14 in Table 7.3) divergence.
Maybe it caused by jacobian of measurement model H or Jacobian of Motion model g(Predict of state covariance)