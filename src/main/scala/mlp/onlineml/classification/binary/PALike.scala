package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector, max}

/**
  * A base class of Passive-Aggressive (PA) algorithms
  *
  * @param w weight vector
  * @see [[http://www.jmlr.org/papers/volume7/crammer06a/crammer06a.pdf Online Passive-Aggressive Algorithms]]
  */
abstract class PALike protected (w: DenseVector[Double])
  extends LinearClassifier(w, DenseMatrix.eye[Double](w.size)) {

  override protected def e(x: DenseVector[Double]): Double = 1.0

  override protected def beta(x: DenseVector[Double], y: Label): Double = 0.0

  /**
    * Calculate hinge loss
    *
    * @param x training sample
    * @param y label for training sample
    * @return hinge loss
    */
  protected def calcHingeLoss(x: DenseVector[Double], y: Label) =
    max(0.0, 1.0 - y.value * (w.t * x))
}
