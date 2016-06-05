package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * PA-II
  *
  * @param w weight vector
  * @param c aggressiveness parameter
  */
class PA_II private(w: DenseVector[Double], val c: Double) extends PALike(w) {
  require(c >= 0.0)

  /**
    * Constructor
    *
    * @param numDimensions num of dimensions
    * @param c aggressiveness parameter (hyper parameter)
    * @return
    */
  def this(numDimensions: Int, c: Double) = this(DenseVector.zeros[Double](numDimensions), c)

  override protected def alpha(x: DenseVector[Double], y: Label): Double =
    calcHingeLoss(x, y) / (x.t * x + 1.0 / (2 * c))

  override protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier = new PA_II(w, c)

  override def name = super.name + s"($c)"
}
