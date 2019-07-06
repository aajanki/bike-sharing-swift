import TensorFlow
import CSVImporter

func loadCsv(_ path: String) -> [[String: String]] {
    let importer = CSVImporter<[String: String]>(path: path)
    return importer.importRecords(structure: { (headerValues) -> Void in () },
                                  recordMapper: { $0 })
}

func parseAsFloatOrZero(_ v: String?) -> Float {
    v.flatMap(Float.init) ?? 0
}

func buildFeatures(_ csvData: [[String: String]]) -> Tensor<Float> {
    let rowCount = csvData.count
    let values = csvData.flatMap { (row) -> [Float] in
        [
          parseAsFloatOrZero(row["workingday"]),
          parseAsFloatOrZero(row["holiday"]),
          parseAsFloatOrZero(row["temp"]),
          parseAsFloatOrZero(row["atemp"]),
          parseAsFloatOrZero(row["hum"]),
          parseAsFloatOrZero(row["windspeed"]),
        ]
    }
    return Tensor<Float>(shape: [rowCount, 6], scalars: values)
}

func buildLabels(_ csvData: [[String: String]]) -> Tensor<Float> {
    Tensor<Float>(
      csvData.map { (row) -> Float in
          parseAsFloatOrZero(row["cnt"])
      })
}

func loadData(_ dataFile: String) -> (Tensor<Float>, Tensor<Float>) {
    let csvData = loadCsv(dataFile)
    let feat = buildFeatures(csvData)
    let labels = buildLabels(csvData)
    return (feat, labels)
}

print("Loading data")
print(loadData("data/day.csv"))
