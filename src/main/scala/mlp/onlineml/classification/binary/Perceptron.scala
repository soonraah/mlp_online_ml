package mlp.onlineml.classification.binary
import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Perceptron
  *
  * @param w weight vector
  * @see [[http://www.sciencedirect.com/science/article/pii/0893608094900914 The perceptron: A probabilistic model for information storage and organization in the brain]]
  */
class Perceptron private (w: DenseVector[Double]) extends LinearClassifier(w, DenseMatrix.eye[Double](w.size)) {
  /**
    * Constructor
    *
    * @param numDimensions num of dimensions
    * @return
    */
  def this(numDimensions: Int) = this(DenseVector.rand[Double](numDimensions))

  override protected def e(x: DenseVector[Double]): Double = 0.0

  override protected def alpha(x: DenseVector[Double], y: Label): Double = 1.0

  override protected def beta(x: DenseVector[Double], y: Label): Double = 0.0

  override protected def create(w: DenseVector[Double],
                                sigma: DenseMatrix[Double]): LinearClassifier = new Perceptron(w)
}
