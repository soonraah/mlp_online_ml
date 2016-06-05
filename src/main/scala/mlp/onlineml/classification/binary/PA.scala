package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Passive-Aggressive (PA)
  *
  * @param w weight vector
  * @see [[http://www.jmlr.org/papers/volume7/crammer06a/crammer06a.pdf Online Passive-Aggressive Algorithms]]
  */
class PA private(w: DenseVector[Double]) extends PALike(w) {
  /**
    * Constructor
    *
    * @param numDimensions num of dimensions
    * @return
    */
  def this(numDimensions: Int) = this(DenseVector.zeros[Double](numDimensions))

  override protected def alpha(x: DenseVector[Double], y: Label): Double = calcHingeLoss(x, y) / (x.t * x)

  override protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier = new PA(w)
}
