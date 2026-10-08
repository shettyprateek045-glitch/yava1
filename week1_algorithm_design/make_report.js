const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, convertInchesToTwip,
} = require("docx");

const PRIMARY = "2C5F2D";
const ACCENT = "C68A2E";
const MUTED = "6B7280";
const CODE_BG = "F2F2F2";

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, bold: true, color: PRIMARY, size: 28 })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 },
    children: [new TextRun({ text, bold: true, color: ACCENT, size: 24 })] });
}
function body(text, opts = {}) {
  return new Paragraph({ spacing: { after: 160, line: 300 },
    children: [new TextRun({ text, size: 22, ...opts })] });
}
function bullet(text) {
  return new Paragraph({ bullet: { level: 0 }, spacing: { after: 90, line: 280 },
    children: [new TextRun({ text, size: 22 })] });
}
function codeBlock(lines) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" },
      left: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" },
      right: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" },
    },
    rows: [new TableRow({ children: [new TableCell({
      shading: { type: ShadingType.CLEAR, fill: CODE_BG },
      margins: { top: 150, bottom: 150, left: 150, right: 150 },
      children: lines.map((l) => new Paragraph({
        spacing: { after: 0, line: 260 },
        children: [new TextRun({ text: l || " ", font: "Consolas", size: 19, color: "1A1D29" })],
      })),
    })] })],
  });
}
function img(path, widthIn, heightIn) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path),
      transformation: { width: widthIn * 96, height: heightIn * 96 } })] });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
    children: [new TextRun({ text, italics: true, size: 19, color: MUTED })] });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1000, bottom: 1000, left: 1100, right: 1100 } } },
    children: [
      new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "WEEK 1 TASK REPORT", bold: true, size: 20, color: ACCENT })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Algorithm Design and Data Preprocessing Strategy", bold: true, size: 36, color: PRIMARY })] }),
      new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "Computer Vision Internship Program \u2014 Prateek, RNS Institute of Technology", size: 21, color: MUTED })] }),

      h1("1. Objective"),
      body("This report develops a complete strategy for a computer vision project: identifying the scope of the problem, designing a preprocessing and augmentation pipeline, proposing a model algorithm, and documenting the reasoning behind each design decision, before any training takes place in Week 2."),

      h1("2. Scope of the Computer Vision Problem"),
      body("The chosen problem is image classification: given an input image, predict which of a fixed set of classes it belongs to. To keep the full pipeline (data generation, preprocessing, model design, training, optimization and deployment) runnable end-to-end and fully reproducible without depending on large external dataset downloads, a synthetic 4-class shape-classification task is used as the concrete instance of this problem:"),
      bullet("Classes: circle, square, triangle, star"),
      bullet("Images: 64x64 RGB, randomized position, size, rotation, color and additive noise per sample"),
      bullet("Why synthetic: perfectly balanced classes, unlimited labeled samples, zero licensing/download concerns, and full control over difficulty (noise level, shape overlap) for testing the pipeline's robustness"),
      body("The same pipeline generalizes directly to a real-world image classification dataset; only the data-loading step would change."),

      h1("3. Data Preprocessing and Augmentation Strategy"),
      h2("3.1 Preprocessing steps"),
      bullet("Resizing: every image is resized to a fixed 32x32 input resolution, trading some spatial detail for significantly faster training and inference"),
      bullet("Normalization: pixel values are scaled from [0, 255] to [0, 1] to keep gradients well-scaled during training"),
      bullet("Channel ordering: images are converted to (Channels, Height, Width) tensor format, the standard layout for convolutional layers"),
      h2("3.2 Augmentation strategy"),
      bullet("Random rotation of the shape within the frame, so the model does not overfit to a single orientation"),
      bullet("Additive Gaussian noise, simulating sensor noise and camera-quality variation"),
      bullet("Randomized position, size and color per sample, which the synthetic generator applies directly during data creation"),
      body("Noise reduction was intentionally not applied as a preprocessing step for this task: the Gaussian noise injected is small relative to shape size, and keeping it lets the trained model's accuracy honestly reflect performance on moderately noisy images, rather than only ever seeing cleaned input."),

      h1("4. Flowcharts"),
      h2("4.1 Data preprocessing pipeline"),
      img("diagrams/data_preprocessing_flowchart.png", 4.2, 7.0),
      caption("Figure 1. End-to-end data preprocessing pipeline, from raw image to model-ready tensor."),
      h2("4.2 Proposed algorithm flow"),
      img("diagrams/algorithm_flowchart.png", 4.2, 9.1),
      caption("Figure 2. Proposed CNN architecture, layer by layer."),

      h1("5. Proposed Algorithm"),
      body("A compact Convolutional Neural Network (CNN) is proposed: two convolution + ReLU + max-pooling blocks for hierarchical feature extraction, followed by two dense (fully-connected) layers for classification. This architecture was chosen over a deeper network because:"),
      bullet("The input resolution (32x32) and problem complexity (4 visually distinct classes) do not require a very deep network to separate classes well"),
      bullet("A shallow network trains and runs inference faster, directly supporting the real-time performance goals evaluated in Week 3"),
      bullet("Two conv blocks are enough to move from raw pixels to edge/shape-level features; a third block was not evaluated; two blocks were chosen as sufficient for this problem's complexity and to keep latency low"),
      body("Full layer-by-layer detail is in Figure 2 above; the implementation lives in week2_model_training/model.py."),

      h1("6. Pseudocode"),
      h2("6.1 Preprocessing pseudocode"),
      codeBlock([
        "function preprocess_image(raw_image, training_mode):",
        "    image = resize(raw_image, target_size = (32, 32))",
        "    image = normalize(image, scale = 1/255.0)",
        "    if training_mode:",
        "        image = random_rotation(image, max_angle = 180)",
        "        image = add_gaussian_noise(image, std = 12)",
        "    tensor = transpose(image, to_format = \"CHW\")",
        "    return tensor",
        "",
        "function build_dataset(samples_per_class):",
        "    dataset = []",
        "    for class_label in CLASSES:",
        "        for i in range(samples_per_class):",
        "            raw_image = generate_or_load_image(class_label)",
        "            tensor = preprocess_image(raw_image, training_mode = True)",
        "            dataset.append((tensor, class_label))",
        "    shuffle(dataset)",
        "    return split_train_val_test(dataset, ratios = (0.8, 0.1, 0.1))",
      ]),
      h2("6.2 Training loop pseudocode"),
      codeBlock([
        "function train(model, train_data, val_data, epochs, learning_rate):",
        "    for epoch in range(epochs):",
        "        shuffle(train_data)",
        "        for batch in minibatches(train_data, batch_size = 32):",
        "            x, y = batch",
        "            logits = model.forward(x)",
        "            loss, gradient = cross_entropy_loss(logits, y)",
        "            model.backward(gradient, learning_rate)",
        "",
        "        val_accuracy = evaluate(model, val_data)",
        "        log(epoch, loss, val_accuracy)",
        "",
        "    return model",
      ]),

      h1("7. Design Rationale"),
      bullet("Accuracy vs. speed: a shallow 2-block CNN was chosen specifically so Week 3's optimization work (pruning, quantization) has a realistic, already-lean baseline to improve on, rather than optimizing an oversized network"),
      bullet("Real-time performance: 32x32 input and a small parameter count (~67K parameters, confirmed in Week 2) keep per-image inference in the low milliseconds, suitable for interactive or near-real-time use"),
      bullet("Augmentation choice: rotation and noise were prioritized over more aggressive augmentations (e.g. heavy occlusion) because the target deployment (Week 4 integration) expects reasonably clean, single-shape input images"),
      bullet("Synthetic data: removes dataset licensing and download-size concerns entirely while still exercising every real stage of a CV pipeline"),

      h1("8. Conclusion"),
      body("This design and preprocessing strategy forms the blueprint for Week 2, where the proposed CNN is implemented and trained on real generated data, and real training metrics replace the design-stage estimates referenced here."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Week1_Report_Algorithm_Design.docx", buf);
  console.log("Week 1 report written");
});
