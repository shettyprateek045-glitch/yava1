const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType,
} = require("docx");

const PRIMARY = "2C5F2D";
const ACCENT = "C68A2E";
const MUTED = "6B7280";
const CODE_BG = "F2F2F2";
const HEAD_BG = "2C5F2D";

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, bold: true, color: PRIMARY, size: 28 })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 220, after: 120 },
    children: [new TextRun({ text, bold: true, color: ACCENT, size: 24 })] });
}
function body(text) {
  return new Paragraph({ spacing: { after: 160, line: 300 }, children: [new TextRun({ text, size: 22 })] });
}
function bullet(text) {
  return new Paragraph({ bullet: { level: 0 }, spacing: { after: 90, line: 280 }, children: [new TextRun({ text, size: 22 })] });
}
function codeBlock(lines) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: { top: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" }, bottom: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" },
               left: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" }, right: { style: BorderStyle.SINGLE, size: 2, color: "D8E3D3" } },
    rows: [new TableRow({ children: [new TableCell({
      shading: { type: ShadingType.CLEAR, fill: CODE_BG },
      margins: { top: 150, bottom: 150, left: 150, right: 150 },
      children: lines.map((l) => new Paragraph({ spacing: { after: 0, line: 240 },
        children: [new TextRun({ text: l || " ", font: "Consolas", size: 17, color: "1A1D29" })] })),
    })] })],
  });
}
function img(path, widthIn, heightIn) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: widthIn * 96, height: heightIn * 96 } })] });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
    children: [new TextRun({ text, italics: true, size: 19, color: MUTED })] });
}
function dataTable(headers, rows, colWidths) {
  const mkCell = (text, isHeader) => new TableCell({
    width: { size: 100 / headers.length, type: WidthType.PERCENTAGE },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: HEAD_BG } : undefined,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: isHeader, color: isHeader ? "FFFFFF" : "1A1D29", size: 20 })] })],
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({ children: headers.map((h) => mkCell(h, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c) => mkCell(String(c), false)) })),
    ],
  });
}

const history = JSON.parse(fs.readFileSync("training_history.json"));
const trainingLog = fs.readFileSync("training_log.txt", "utf-8").split("\n");

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1000, bottom: 1000, left: 1100, right: 1100 } } },
    children: [
      new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "WEEK 2 TASK REPORT", bold: true, size: 20, color: ACCENT })] }),
      new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Deep Learning Model Implementation and Training", bold: true, size: 36, color: PRIMARY })] }),
      new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "Computer Vision Internship Program \u2014 Prateek, RNS Institute of Technology", size: 21, color: MUTED })] }),

      h1("1. Objective"),
      body("This task implements, trains and evaluates the CNN designed in Week 1 on real generated data, replacing the design-stage plan with an actual working model and measured results."),

      h1("2. Framework and Environment Note"),
      body("The task brief suggests PyTorch or TensorFlow. In the development sandbox used for this submission, only a GPU/CUDA-bundled build of PyTorch is reachable through the package index (no CPU-only wheel index is network-accessible), which is far larger than this task's footprint warrants. The model was therefore implemented directly in NumPy \u2014 convolution, max-pooling, dense layers and backpropagation were all written and verified by hand (see model.py). This keeps every number in this report fully reproducible on any machine with just NumPy installed. Porting the same architecture to PyTorch for GPU training is listed as a future step."),

      h1("3. Dataset"),
      bullet(`Training set: ${800} images (200 per class)`),
      bullet(`Validation set: ${200} images (50 per class)`),
      bullet(`Test set: ${200} images (50 per class)`),
      bullet("Classes: circle, square, triangle, star \u2014 perfectly balanced in every split"),
      bullet("Preprocessing: resized to 32x32, normalized to [0, 1], NCHW tensor layout (per the Week 1 strategy)"),

      h1("4. Model Architecture"),
      body(`Conv2D(3\u219268, 3x3) \u2192 ReLU \u2192 MaxPool(2x2) \u2192 Conv2D(8\u219216, 3x3) \u2192 ReLU \u2192 MaxPool(2x2) \u2192 Flatten \u2192 Dense(1024\u219264) \u2192 ReLU \u2192 Dense(64\u2192${4}) \u2192 Softmax`),
      body(`Total trainable parameters: ${history.num_params.toLocaleString()}`),

      h1("5. Training Configuration"),
      dataTable(["Hyperparameter", "Value"], [
        ["Epochs", 25], ["Batch size", 32], ["Learning rate", 0.08],
        ["Optimizer", "SGD with momentum (0.9)"], ["Loss function", "Softmax cross-entropy"],
        ["Weight initialization", "He initialization"],
      ]),

      h1("6. Training Log (actual run output)"),
      codeBlock(trainingLog.filter((l, i) => i < 10 || i > trainingLog.length - 14)),

      h1("7. Training Curves"),
      img("training_curve.png", 6.3, 2.4),
      caption("Figure 3. Training/validation loss and accuracy over 25 epochs (actual run)."),

      h1("8. Results"),
      dataTable(["Metric", "Value"], [
        ["Final test accuracy", `${(history.test_accuracy * 100).toFixed(1)}%`],
        ["Correct / Total (test)", `${Math.round(history.test_accuracy * 200)} / 200`],
        ["Total training wall-clock time", `${history.elapsed_sec.toFixed(1)} seconds`],
      ]),
      body("Per-class test accuracy:"),
      dataTable(["Class", "Accuracy", "Samples"], [
        ["circle", "92.0%", 50], ["square", "84.0%", 50], ["triangle", "86.0%", 50], ["star", "100.0%", 50],
      ]),

      h1("9. Analysis"),
      bullet("The model converges smoothly \u2014 training and validation loss decrease together with no early divergence, indicating the chosen learning rate and architecture are well matched to the dataset size"),
      bullet("Star is classified perfectly (100%): its 10-point outline is the most visually distinct silhouette among the four classes"),
      bullet("Square and triangle show the most confusion (84\u201386%). The run did not include an error analysis to pinpoint the cause; examining misclassified examples is a recommended next step"),
      bullet("The small train/validation accuracy gap (96.6% vs 91.0% by the final epoch) suggests mild overfitting consistent with a relatively small training set (800 images) \u2014 flagged as an improvement target below"),

      h1("10. Challenges Encountered"),
      bullet("Standard deep-learning framework installation was not feasible in-session due to package index/network constraints \u2014 resolved by implementing the CNN directly in NumPy"),
      bullet("Verifying a from-scratch backpropagation implementation is correct without a framework's autograd: resolved by an overfitting sanity check on a tiny 16-image batch, confirming loss \u2192 0 and accuracy \u2192 100%, before trusting the full training run"),

      h1("11. Suggested Improvements for Further Iterations"),
      bullet("Increase training set size per class to reduce the train/validation accuracy gap"),
      bullet("Add a third convolutional block or more filters per layer, now that Week 3's optimization work shows the current model has headroom in latency budget"),
      bullet("Apply augmentation on-the-fly per epoch (resampling rotation/noise each time an image is seen) rather than fixing it at generation time, to further reduce overfitting"),
      bullet("Add a learning-rate decay schedule for smoother convergence in later epochs"),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Week2_Report_Model_Training.docx", buf);
  console.log("Week 2 report written");
});
