package mlp.reader
import java.io.{BufferedReader, IOException}

import breeze.linalg.DenseVector
import mlp.onlineml.classification.binary.Label

/**
  * Stream reader for Breast Cancer Wisconsin (Diagnostic) Data Set
  *
  * @param br buffered reader of the file wdbc.data
  * @see [[http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29 Breast Cancer Wisconsin (Diagnostic) Data Set]]
  */
class WDBCReader(br: BufferedReader) extends StreamVectorReader {

  override val numDimensions = 30

  override def close(): Unit = {
    if (br != null) {
      try {
        br.close()
      } catch {
        case e: IOException => throw new IOException("Failed to close buffered reader", e)
      }
    }
  }

  override def read(): Option[(Label, DenseVector[Double])] = {
    try {
      val line = br.readLine()
      if (line == null) return None
      val items = line.split(",")
      val label = if (items(1) == "M") Label(true) else Label(false)  // M: malignant, B: benign
      val xArray = items.drop(2).map(_.toDouble)
      if (xArray.length != numDimensions) throw new IOException(s"Num of data values must be $numDimensions")
      Some((label, new DenseVector[Double](xArray)))
    } catch {
      case e: Throwable => throw new IOException("Failed to read line", e)
    }
  }
}
