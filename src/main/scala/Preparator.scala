package org.template.classification

import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext

class PreparedData(
  val labeledPoints: DataFrame
) extends Serializable

class Preparator extends PPreparator[TrainingData, PreparedData] {
  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    new PreparedData(trainingData.labeledPoints.toDF)
  }
}
