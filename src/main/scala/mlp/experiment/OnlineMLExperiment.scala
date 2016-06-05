package mlp.experiment

import java.io._

import mlp.onlineml.classification.binary._
import mlp.reader.StreamVectorReader

import scala.annotation.tailrec


class OnlineMLExperiment {
  val DELIMITER = "\t"

  def run(svr: StreamVectorReader, bw: BufferedWriter, classifiers: Seq[LinearClassifier]): Unit = {
    @tailrec
    def rec(t: Int, classifiers: Seq[LinearClassifier]): Unit = {
      svr.read() match {
        case Some((y, x)) =>
          val predicted = classifiers.zipWithIndex.map { t => t._1.classify(x) }
          val lineItems = t.toString +: y.value.toString +: predicted.map(_.value.toString)
          bw.write(lineItems.mkString(DELIMITER) + "\n")
          rec(t + 1, classifiers.zipWithIndex.map { t => t._1.train(x, y) })
        case _ =>
      }
    }

    try {
      val header = "t" +: "correct" +: classifiers.map(_.name)
      bw.write(header.mkString(DELIMITER) + "\n")
      rec(0, classifiers)
    } catch {
      case e: Throwable => throw new RuntimeException("Failed to run online ML experiment: ", e)
    }
  }
}
