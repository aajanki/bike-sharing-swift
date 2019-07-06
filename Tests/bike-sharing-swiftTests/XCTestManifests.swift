import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(bike_sharing_swiftTests.allTests),
    ]
}
#endif
