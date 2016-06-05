package mlp.onlineml.classification.binary
import breeze.linalg.{DenseMatrix, DenseVector, max}

/**
  * Confidence Weighted Learning (CW)
  *
  * @param w weight vector
  * @param sigma covariance matrix
  * @param eta confidence hyper parameter
  * @see [[http://webee.technion.ac.il/people/koby/publications/paper_nips08_std.pdf Exact Convex Confidence-Weighted Learning]]
  */
class CW private (w: DenseVector[Double], sigma: DenseMatrix[Double], eta: Double) extends CWLike(w, sigma, eta) {
  /**
    * Constructor
    *
    * @param numDimensions num of dimensions
    * @param a initial sigma coefficient (hyper parameter)
    * @param eta confidence hyper parameter
    */
  def this(numDimensions: Int, a: Double, eta: Double) = {
    this(DenseVector.zeros[Double](numDimensions), DenseMatrix.eye[Double](numDimensions) * a, eta)
    require(a > 0.0)
  }

  override protected def alpha(x: DenseVector[Double], y: Label): Double = max(0.0, gamma(x, y))

  override protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier =
    new CW(w, sigma, eta)

  override def name = super.name + s"($eta)"
}
