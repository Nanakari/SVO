# Stage Plan

The project is implemented in inspection gates. Each stage should be reviewed before the next one starts.

| Stage | Scope | Status |
| --- | --- | --- |
| 1 | Project structure and README | Done |
| 2 | Base Caption and POPE inference code | Done |
| 3 | Object extraction and risk scoring code | Done |
| 4 | GroundingDINO verification code | Done |
| 5 | SVO conservative revision logic | Done |
| 6 | CHAIR / POPE / efficiency / false-correction evaluation | Done |
| 7 | Automatic result export scripts | Done |

Completion rules:

- Do not fabricate experimental values.
- Do not tune thresholds on test data.
- Keep VCD and OPERA as TODO interfaces unless the actual pipelines are connected.
- Keep model, detector, and dataset modules replaceable.
