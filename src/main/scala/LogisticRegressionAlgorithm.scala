package org.template.classification

import grizzled.slf4j.Logger
import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import io.prediction.controller.PersistentModel
import io.prediction.controller.PersistentModelLoader
import io.prediction.controller.Utils
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SQLContext

case class AlgorithmParams(
  lambda: Double
) extends Params

class ModelWrapper(val sc: SparkContext, val model: LogisticRegressionModel)
  extends PersistentModel[AlgorithmParams] {
  def save(id: String, params: AlgorithmParams, sc: SparkContext): Boolean = {
    Utils.save(id, model)
    true
  }
}

object ModelWrapper extends PersistentModelLoader[AlgorithmParams, ModelWrapper] {
  def apply(id: String, params: AlgorithmParams, sc: Option[SparkContext]): ModelWrapper = {
    new ModelWrapper(sc.get, Utils.load(id).asInstanceOf[LogisticRegressionModel])
  }
}

class LogisticRegressionAlgorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, ModelWrapper, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): ModelWrapper = {
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")

    val lr = new LogisticRegression()
    lr.setMaxIter(10).setRegParam(0.01)
    new ModelWrapper(sc, lr.fit(data.labeledPoints))
  }

  def predict(model: ModelWrapper, query: Query): PredictedResult = {
    val sc = model.sc
    val sqlc = new SQLContext(sc)
    import sqlc.implicits._
    val q = sc.parallelize(Seq(LabeledPoint(0.0, Vectors.dense(query.features)))).toDF
    val label = model.model.transform(q).select("prediction").collect().head.getDouble(0)
    new PredictedResult(label)
  }
}
