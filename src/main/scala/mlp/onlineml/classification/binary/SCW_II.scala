package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector, max}
import breeze.numerics._

/**
  * SCW-II (Soft Confidence-Weighted Learning -II)
  *
  * @param w weight vector
  * @param sigma covariance matrix
  * @param c parameter to trade off the passiveness and aggressiveness (hyper parameter)
  * @param eta confidence hyper parameter
  * @see [[http://icml.cc/2012/papers/86.pdf Exact Soft Confidence-Weighted Learning]]
  */
class SCW_II private(w: DenseVector[Double], sigma: DenseMatrix[Double], c: Double, eta: Double)
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

  override protected def alpha(x: DenseVector[Double], y: Label): Double = {
    val v = calcV(x)
    val m = calcM(x, y)
    val n = v + 1 / (2 * c)
    val gamma = phi * sqrt(pow(phi * m * v, 2) + 4 * n * v * (n + v * pow(phi, 2)))
    max(0.0, (-1 * (2 * m * n + pow(phi, 2) * m * v) + gamma) / (2.0 * (pow(n, 2) + n * v * pow(phi, 2))))
  }

  override protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier =
    new SCW_II(w, sigma, c, eta)

  override def name = super.name + s"($c,$eta)"
}
