package mlp.reader

import breeze.linalg.DenseVector
import mlp.onlineml.classification.binary.Label

trait StreamVectorReader {
  /** num of dimensions for vectors */
  val numDimensions: Int

  /**
    * Read data source and return vectorized data
    *
    * @return a vectorized sample (None if end of data source)
    */
  def read(): Option[(Label, DenseVector[Double])]

  /**
    * Close reader
    */
  def close(): Unit
}
