package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector, max}

/**
  * Adaptive Regularization of Weight Vectors (AROW)
  *
  * @param w weight vector
  * @param sigma covariance matrix
  * @param r regularization term (hyper parameter)
  * @see [[http://papers.nips.cc/paper/3848-adaptive-regularization-of-weight-vectors.pdf Adaptive Regularization of Weight Vectors]]
  */
class AROW private (w: DenseVector[Double], sigma: DenseMatrix[Double], r: Double) extends LinearClassifier(w, sigma) {
  require(r > 0.0)

  /**
    * Constructor
    *
    * @param numDimensions num of dimensions
    * @param r regularization term (hyper parameter)
    * @return
    */
  def this(numDimensions: Int, r: Double) =
    this(DenseVector.zeros[Double](numDimensions), DenseMatrix.eye[Double](numDimensions), r)

  private def mu = w

  override protected def e(x: DenseVector[Double]): Double = 1.0

  override protected def alpha(x: DenseVector[Double], y: Label): Double =
    max(0.0, 1.0 - y.value * (x.t * mu)) * beta(x, y)

  override protected def beta(x: DenseVector[Double], y: Label): Double =
    1.0 / ((x.t * sigma * x) + r)

  override protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier =
    new AROW(w, sigma, r)

  override def name = super.name + s"($r)"
}
