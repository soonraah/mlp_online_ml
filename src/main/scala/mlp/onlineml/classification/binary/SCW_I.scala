package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector, max, min}

/**
  * SCW-I (Soft Confidence-Weighted Learning -I)
  *
  * @param w weight vector
  * @param sigma covariance matrix
  * @param c parameter to trade off the passiveness and aggressiveness (hyper parameter)
  * @param eta confidence hyper parameter
  * @see [[http://icml.cc/2012/papers/86.pdf Exact Soft Confidence-Weighted Learning]]
  */
class SCW_I private(w: DenseVector[Double], sigma: DenseMatrix[Double], c: Double, eta: Double)
  extends CWLike(w, sigma, eta) {
  require(c > 0.0)

  /**
    * Constructor
    *
    * @param numDimensions num of dimensions
    * @param c parameter to trade off the passiveness and aggressiveness (hyper parameter)
    * @param eta confidence hyper parameter
    * @return
    */
  def this(numDimensions: Int, c: Double, eta: Double) =
    this(DenseVector.zeros[Double](numDimensions), DenseMatrix.eye[Double](numDimensions), c, eta)

  override protected def alpha(x: DenseVector[Double], y: Label): Double = min(c, max(0.0, gamma(x, y)))

  override protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier =
    new SCW_I(w, sigma, c, eta)

  override def name = super.name + s"($c,$eta)"
}
