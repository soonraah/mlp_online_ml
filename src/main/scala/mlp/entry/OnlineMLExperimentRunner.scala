package mlp.entry

import java.io._

import mlp.experiment.OnlineMLExperiment
import mlp.onlineml.classification.binary._
import mlp.reader.{StreamVectorReader, WDBCReader}

object OnlineMLExperimentRunner {
  def main(args: Array[String]) {
    // download it from http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
    val wdbcFilename = "data/wdbc.data"
    val br = new BufferedReader(new InputStreamReader(new FileInputStream(wdbcFilename)))
    val svr: StreamVectorReader = new WDBCReader(br)

    val outputFilename = "out.tsv"
    val bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFilename)))

    val numDimensions = svr.numDimensions

    // here, hyper parameters are tuned lightly
    val classifiers: Seq[LinearClassifier] = Seq(
      new Perceptron(numDimensions),
      new PA(numDimensions),
      new PA_I(numDimensions, 1.0),
      new PA_II(numDimensions, 1.0),
      new CW(numDimensions, 1.0, 0.64),
      new AROW(numDimensions, 0.01),
      new SCW_I(numDimensions, 0.04, 0.7),
      new SCW_II(numDimensions, 0.007, 0.94)
    )

    // run
    try {
      new OnlineMLExperiment().run(svr, bw, classifiers)
    } finally {
      svr.close()
      bw.close()
    }
  }
}
