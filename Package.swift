// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MetalHashmlp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "MetalHashmlp",
            targets: ["MetalHashmlp"]),
        .executable(
            name: "LearningAnImage",
            targets: ["Examples"])
    ],
    targets: [
        .target(
            name: "MetalHashmlp",
            resources: [
                .copy("metal")
            ]),
        .executableTarget(
            name: "Examples",
            dependencies: ["MetalHashmlp"],
            resources: [
                .copy("albert.jpg"),
                .copy("metal")
            ])
    ]
)
