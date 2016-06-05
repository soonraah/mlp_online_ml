package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.stats.distributions.Gaussian

/**
  * base class of Confidence Weighted Learning (CW) algorithms
  *
  * @param w weight vector
  * @param sigma covariance matrix
  * @param eta confidence hyper parameter
  * @see [[http://webee.technion.ac.il/people/koby/publications/paper_nips08_std.pdf Exact Convex Confidence-Weighted Learning]]
  */
abstract class CWLike protected (w: DenseVector[Double], sigma: DenseMatrix[Double], val eta: Double) extends LinearClassifier(w, sigma) {
  require(0.5 <= eta && eta <= 1.0)

  protected val phi = Gaussian(0.0, 1.0).icdf(eta)
  protected val psi = 1 + (phi * phi) / 2
  protected val xi = 1 + phi * phi

  protected def mu = w
  protected def calcV(x: DenseVector[Double]) = x.t * sigma * x
  protected def calcM(x: DenseVector[Double], y: Label) = y.value * (mu.t * x)

  protected def gamma(x: DenseVector[Double], y: Label): Double = {
    val v = calcV(x)
    val m = calcM(x, y)
    (1.0 / v * xi) * (-1 * m * psi + sqrt(pow(m, 2) * pow(phi, 4) / 4.0 + v * pow(phi, 2) * xi))
  }

  override protected def e(x: DenseVector[Double]): Double = phi * sqrt(calcV(x))

  override protected def beta(x: DenseVector[Double], y: Label): Double = {
    val v = calcV(x)
    val alphaValue = alpha(x, y)
    val avp = alphaValue * v * phi
    val u = 0.25 * pow(-1 * avp + sqrt(pow(avp, 2) + 4 * v), 2)
    alphaValue * phi / (sqrt(u) + avp)
  }
}
