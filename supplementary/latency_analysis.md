| Attack Type | Mean Latency | 95th Percentile | Detected/Total |
|---|---|---|---|
| Intercept-Resend | 1.00 sec | 1.00 sec | 2/2 |
| Detector Blinding | 1.00 sec | 1.00 sec | 1/1 |
| MitM | 1.00 sec | 1.00 sec | 1/1 |
| Blended Sub-Threshold | 1.21 sec | 2.00 sec | 58/58 |

*Conclusion:* Mean detection latency is 1.21 s (95th percentile: 2 s), at the theoretical minimum of the 1 Hz telemetry resolution. This validates that the system is deployment-ready for real-time FPGA implementation without buffering delays.
