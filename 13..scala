import org.apache.spark.sql.SparkSession

object SimpleSpark {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession.builder
            .appName("Simple Spark Program")
            .master("local[*]")
            .getOrCreate()

        val data = Seq(
            ("Shravan", 20),
            ("Rahul", 21),
            ("Amit", 22)
        )

        import spark.implicits._

        val df = data.toDF("Name", "Age")

        df.show()

        spark.stop()
    }
}
