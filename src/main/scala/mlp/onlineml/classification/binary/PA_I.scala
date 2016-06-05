package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector, min}

/**
  * PA-I
  *
  * @param w weight vector
  * @param c aggressiveness parameter (hyper parameter)
  */
class PA_I private(w: DenseVector[Double], val c: Double) extends PALike(w) {
  require(c >= 0.0)

  /**
    * Constructor
    *
    * @param numDimensions num of dimensions
    * @param c aggressiveness parameter (hyper parameter)
    * @return
    */
  def this(numDimensions: Int, c: Double) = this(DenseVector.zeros[Double](numDimensions), c)

  override protected def alpha(x: DenseVector[Double], y: Label): Double = min(c, calcHingeLoss(x, y) / (x.t * x))

  override protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier = new PA_I(w, c)

  override def name = super.name + s"($c)"
}
