# Quantized Model Candidate Admission Service

Stateful two-phase candidate-admission service providing `POST /quantize` endpoint for freezing model candidates (`phase: freeze`) and selecting admitted quantized artifacts (`phase: select`).

## API Endpoint
- `POST /quantize`: Freeze candidate artifacts or select admitted quantized model.
- `GET /`: Health check endpoint.
