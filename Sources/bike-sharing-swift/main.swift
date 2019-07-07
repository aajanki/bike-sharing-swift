import Foundation
import TensorFlow
import CSVImporter

struct Samples {
    var features: Tensor<Float>
    var labels: Tensor<Float>
}

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

func loadData(_ dataFile: String) -> Samples {
    let csvData = loadCsv(dataFile)
    let feat = buildFeatures(csvData)
    let labels = buildLabels(csvData)
    return Samples(features: feat, labels: labels)
}

func trainTestSplit(samples: Samples, testProportion: Double = 0.2) -> (Samples, Samples) {
    // use the last part of the time series as test data
    let count = samples.features.shape[0]
    let testCount = Int(testProportion*Double(count))
    let trainCount = count - testCount

    return (
      train: Samples(features: samples.features[0..<trainCount], labels: samples.labels[0..<trainCount]),
      test: Samples(features: samples.features[trainCount...], labels: samples.labels[trainCount...])
    )
}

struct Regression: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    static let hiddenDim = 16
    var dense1 = Dense<Float>(inputSize: 27, outputSize: hiddenDim, activation: relu)
    var dense2 = Dense<Float>(inputSize: hiddenDim, outputSize: 1, activation: identity)

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        input.sequenced(through: dense1, dense2)
    }
}

let epochCount = 2000
let batchSize = 64
let dataset = loadData("data/day.csv")
let (trainData, testData) = trainTestSplit(samples: dataset)

print("Number of samples: train: \(trainData.features.shape[0]), test: \(testData.features.shape[0])")
print("Number of features: \(trainData.features.shape[1])")

var model = Regression()
let optimizer = Adam(for: model, learningRate: 5e-3)

let batchCount = Int(ceil(Float(trainData.features.shape[0])/Float(batchSize)))
for epoch in 1...epochCount {
    Context.local.learningPhase = .training
    var totalTrainingLoss: Float = 0
    for i in 0..<batchCount {
        let start = i*batchSize
        let batchX = trainData.features[start..<(start + batchSize)]
        let batchY = trainData.labels[start..<(start + batchSize)]
        let (loss, 𝛁model) = model.valueWithGradient { model -> Tensor<Float> in
            let ŷ = model(batchX)
            return meanSquaredError(predicted: ŷ, expected: batchY)
        }

        totalTrainingLoss += loss.scalarized()

        optimizer.update(&model.allDifferentiableVariables, along: 𝛁model)
    }
    let trainingMAE = meanAbsoluteError(predicted: model(trainData.features), expected: trainData.labels)

    Context.local.learningPhase = .inference

    let testPred = model(testData.features)
    let testMAE = meanAbsoluteError(predicted: testPred, expected: testData.labels)

    if epoch % 50 == 0 {
        print("Epoch: \(epoch), train loss: \(totalTrainingLoss), training MAE: \(trainingMAE), test MAE: \(testMAE)")
    }
}
