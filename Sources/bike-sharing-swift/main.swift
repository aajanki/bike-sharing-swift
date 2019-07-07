import Foundation
import TensorFlow
import CSVImporter

func loadCsv(_ path: String) -> [[String: String]] {
    let importer = CSVImporter<[String: String]>(path: path)
    return importer.importRecords(structure: { (headerValues) -> Void in () },
                                  recordMapper: { $0 })
}

func parseAsIntOrZero(_ v: String?) -> Int32 {
    v.flatMap(Int32.init) ?? 0
}

func parseAsFloatOrZero(_ v: String?) -> Float {
    v.flatMap(Float.init) ?? 0
}

func oneHotTensor(_ csvData: [[String: String]], column: String, depth: Int) -> Tensor<Float> {
    let values = Tensor(
      csvData.map { (row) -> Int32 in
          parseAsIntOrZero(row[column])
      })
    return Tensor<Float>.init(oneHotAtIndices: values, depth: depth)
}

func parseYearsSinceFeature(_ csvData: [[String: String]], column: String) -> Tensor<Float> {
    let secondsPerYear: Double = 60*60*24*365
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd"

    let referenceDate = csvData[0][column].flatMap(dateFormatter.date) ?? Date.init(timeIntervalSince1970: 0)
    let yearsSince = csvData.map { (row) -> Float in
        row[column]
          .flatMap(dateFormatter.date)
          .map { (d) -> Float in
              Float(d.timeIntervalSince(referenceDate)/secondsPerYear)
          }
          ?? 0
    }
    
    return Tensor(shape: [yearsSince.count, 1], scalars: yearsSince)
}

func buildFeatures(_ csvData: [[String: String]]) -> Tensor<Float> {
    let rowCount = csvData.count
    let scalars = csvData.flatMap { (row) -> [Float] in
        [
          parseAsFloatOrZero(row["workingday"]),
          parseAsFloatOrZero(row["holiday"]),
          parseAsFloatOrZero(row["temp"]),
          parseAsFloatOrZero(row["atemp"]),
          parseAsFloatOrZero(row["hum"]),
          parseAsFloatOrZero(row["windspeed"]),
        ]
    }
    let scalarFeatures = Tensor<Float>(shape: [rowCount, 6], scalars: scalars)
    let seasons = oneHotTensor(csvData, column: "season", depth: 4)
    let months = oneHotTensor(csvData, column: "mnth", depth: 12)
    let weather = oneHotTensor(csvData, column: "weathersit", depth: 4)
    let yearsSince =  parseYearsSinceFeature(csvData, column: "dteday")

    return scalarFeatures
      .concatenated(with: seasons, alongAxis: 1)
      .concatenated(with: months, alongAxis: 1)
      .concatenated(with: weather, alongAxis: 1)
      .concatenated(with: yearsSince, alongAxis: 1)
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

let (X, y) = loadData("data/day.csv")

print("Number of samples: \(X.shape[0])")
print("Number of features: \(X.shape[1])")

print(X.min(alongAxes: 0))
print(X.max(alongAxes: 0))
