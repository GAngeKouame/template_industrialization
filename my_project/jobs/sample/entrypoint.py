from template_industrialization.common import Job


#getting dbutils




class MachineLearningModel(Job):


    def training_part(self):
        #################### loading different kind of data ########################
        sfOptions = {
            "sfURL": "danone.west-europe.azure.snowflakecomputing.com",
            "sfAccount": "danone",
            "sfUser": "ANGE.KOUAME@EXTERNAL.DANONE.COM",
            "sfRole": "DEV_CBS",
            "sfDatabase": "DEV_CBS",
            "sfSchema": "CBS_DSP",
            "sfWarehouse": "DEV_ANL_WH",
            "pem_private_key": self.dbutils.secrets.get(scope="DAN-EU-T-KVT800-S-DB-SBX", key="demoange")
        }
        dataset = self.spark.read.\
                            format("net.snowflake.spark.snowflake").\
                            options(**sfOptions).option("dbtable", "DATASET").\
                            load()

        technologies = self.spark.read. \
                                 format("net.snowflake.spark.snowflake"). \
                                 options(**sfOptions).option("dbtable", "TECHNOLOGIES"). \
                                 load()
        #####################################################################################
        # renamed column _c0 as index
        technologies = technologies.withColumnRenamed("_c0", "index")

        # join the two dataset on the column index
        general_df = dataset.join(technologies, on=["index"])

        # select desired columns to predict job of missing ones (Diplome, index) and 8th column to the end
        predictors = general_df.select(["Diplome", "index"] + general_df.columns[8:])

        # create numeric values for the Diplome columns elements  by using StringIndexer and call the output Diplome_string_encoded
        stage_string = [StringIndexer(inputCol="Diplome", outputCol="Diplome_string_encoded")]

        # one hot encode the column Diplome_string_encoded and call the output Diplome_one_hot
        stage_one_hot = [OneHotEncoder(inputCol="Diplome_string_encoded", outputCol="Diplome_one_hot")]

        # create a pipeline with previous stages created
        ppl = Pipeline(stages=stage_string + stage_one_hot)

        # transform the dataset predictors by using ppl
        df = ppl.fit(predictors).transform(predictors)

        # the combined output column's name
        assembler = VectorAssembler(
            inputCols = df.columns[2:-2] + [df.columns[-1]], outputCol='features'
        )
        df2 = assembler.transform(df).select("index", "features")
        # take data with missing "metiers"
        features_test = (dataset.filter(col("metier")
                                        .isNull())
                         .drop("Metier")
                         .select("index")
                         .join(df2, on=["index"])
                         .select("index", "features")
                         )

        features_label_training = (dataset.filter(col("metier")
                                                  .isNotNull())
                                   .select("index", "metier")
                                   .join(df2, on=["index"])
                                   .select("index", "features", "metier")
                                   )

        # use it to transform the dataset and select just
        # the output column
        df2 = assembler.transform(df).select("index", "features")

        # take data with missing "metiers"
        features_test = (dataset.filter(col("metier")
                                        .isNull())
                         .drop("Metier")
                         .select("index")
                         .join(df2, on=["index"])
                         .select("index", "features")
                         )

        features_label_training = (dataset.filter(col("metier")
                                                  .isNotNull())
                                   .select("index", "metier")
                                   .join(df2, on=["index"])
                                   .select("index", "features", "metier")
                                   )

        # we can't work with string labels in pyspark ml algorithms so we need to encode them to numeric by using StringIndexer
        stage_string = StringIndexer(inputCol="metier", outputCol="metier_string_encoded")
        new_data = stage_string.fit(features_label_training).transform(features_label_training)
        #predictions
        # divide data in two part a training part and  a test part with a given seed, to have the same distributions
        # set seed to 42 here
        train, self.test = new_data.randomSplit([0.9, 0.1], seed=42)
        for num_trees in range(20, 701, 30):
            with mlflow.start_run():
                # we will use here a logistic regression classifier to try to predict jobs
                # set the number of trees to 60
                # num_trees = 20
                # set max_depth to 3
                max_depth = np.random.choice([3, 5, 8, 10])
                # set feature selection to "auto"
                strategy = np.random.choice(["log2", "sqrt", "auto", "1", "0.6"])
                dt = (RandomForestClassifier(labelCol="metier_string_encoded", featuresCol="features",
                                             numTrees=num_trees,
                                             maxDepth=max_depth,
                                             featureSubsetStrategy=strategy)
                      )

                # fit the model with the train dataset created earlier
                self.model = dt.fit(train)

                # log parameters (register paramters to see them in the mlflow experiments)
                mlflow.log_param("num_trees", num_trees)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("subset_strategy", strategy)
                mlflow.set_tag("column_to_predict", "metier")

        #self.logger.info

    def training_part(self):
        predictions = self.model.transform(self.test)
        metric_names = ["accuracy", "weightedPrecision",
                        "weightedRecall", "logLoss"]
        for elt in metric_names:
            evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                          predictionCol="prediction",
                                                          metricName=elt)
            metric_used = evaluator.evaluate(predictions)
            # log metrics
            mlflow.log_metric(evaluator.getMetricName(), metric_used)

        # log model (register the model used, here a random forest ensemble)
        mlflow.spark.log_model(self.model, "random forest")

if __name__ == "__main__":
    job = SampleJob()
    job.launch()
