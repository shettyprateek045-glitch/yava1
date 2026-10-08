# Computer Vision Internship — Shape Classification Pipeline

Four-week computer vision project, built as a complete, runnable pipeline:
design → model training → optimization → integration and testing.

The task is a 4-class synthetic image classification problem (circle, square,
triangle, star). Synthetic data keeps the full pipeline reproducible without
external dataset downloads.

## Repository layout

| Folder | Contents |
|---|---|
| `week1_algorithm_design/` | Design report (`.docx`), flowchart diagrams, pseudocode |
| `week2_model_training/` | Dataset generator, NumPy CNN, training script, trained checkpoint, training logs and curves, report |
| `week3_optimization/` | Magnitude pruning and int8 quantization, benchmark results and chart, report |
| `week4_integration/` | FastAPI service, integration test suite, test results, architecture diagram, report |

## Reproducing the results

```bash
pip install -r requirements.txt
cd week2_model_training && python3 train.py        # trains CNN, writes checkpoint + logs
cd ../week3_optimization && python3 optimize.py    # prunes, quantizes, benchmarks
cd ../week4_integration  && python3 test_client.py # runs integration tests
cd ../week4_integration  && python3 api.py         # starts the service on :8000
```

## Key results (from the actual runs in these folders)

- **Week 2:** 90.5% test accuracy, 67,252 parameters, 26 s training time
- **Week 3:** int8 quantization cut weight storage by 74.9% for a 1.0-point accuracy drop; pruning (40%) cost 4.0 points
- **Week 4:** 59 of 63 integration tests passed; 56 of 60 live-API predictions correct

## Limitations

- The CNN is implemented in NumPy rather than PyTorch/TensorFlow (framework install was not feasible in the development environment). Porting to PyTorch for GPU training is a listed future step.
- Quantization is simulated (fake-quant); no integer-kernel speedup is claimed.
- Timings are relative comparisons on a single CPU; they are not hardware benchmarks.
- Results are on synthetic data; performance on real photographs is not measured.
