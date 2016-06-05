package mlp.onlineml.classification.binary

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * A base class of binary linear classification
  *
  * @param w weight vector
  * @param sigma covariance matrix
  */
abstract class LinearClassifier protected (val w: DenseVector[Double], val sigma: DenseMatrix[Double]) {
  /**
    * Update criterion
    *
    * @param x training sample
    * @return update criterion
    */
  protected def e(x: DenseVector[Double]): Double

  /**
    * Coefficient to update weight vector
    *
    * @param x training sample
    * @param y label for training sample
    * @return coefficient to update weight vector
    */
  protected def alpha(x: DenseVector[Double], y: Label): Double

  /**
    * Coefficient to update covariance matrix
    *
    * @param x training sample
    * @param y label for training sample
    * @return coefficient to update covariance matrix
    */
  protected def beta(x: DenseVector[Double], y: Label): Double

  /**
    * Online training
    *
    * @param x training sample
    * @param y label for training sample
    * @return updated classifier
    */
  def train(x: DenseVector[Double], y: Label): LinearClassifier = {
    require(x.length == w.length)
    if (y.value * (w.t * x) < e(x)) {
      create(
        w + y.value * alpha(x, y) * (sigma * x),
        sigma - beta(x, y) * (sigma * x * x.t * sigma)
      )
    } else {
      create(w, sigma)
    }
  }

  /**
    * Classify a sample by trained classifier
    *
    * @param x a sample to classify
    * @return predicted label
    */
  def classify(x: DenseVector[Double]): Label = {
    if (w.t * x > 0) Label(true) else Label(false)
  }

  /**
    * Create an updated classifier
    *
    * @param w weight vector
    * @param sigma covariance matrix
    * @return
    */
  protected def create(w: DenseVector[Double], sigma: DenseMatrix[Double]): LinearClassifier

  def name: String = this.getClass.getName.split('.').last
}
