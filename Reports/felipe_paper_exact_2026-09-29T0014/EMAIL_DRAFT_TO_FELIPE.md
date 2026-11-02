# Draft email to Felipe (NOT SENT)

Hi Felipe,

I completed and verified the V1→V1 MLC results using the released 355 × 827 feature matrix for RF, IPCARF and RFLDA.

To apply exactly the same feature construction to V2, I tried to reconstruct the 827 features from the V1 sequences. I can reproduce 462 DCC columns and 80 two-mer/three-mer columns, but the repository and paper do not contain the configuration needed to reproduce the remaining 285 columns.

Could you please share the original Pse-in-One feature-extraction script or command lines, including its version, descriptor parameters, column order, normalization and handling of ambiguous nucleotides?

For the cross-version comparison, I found 48 shared unique sequences. Forty-four have one-to-one correspondence, while four map to multiple V2 identifiers. I already produced a primary analysis using the 44 one-to-one pairs and a sensitivity analysis using all 48 shared sequences. Please let me know whether you would like these results included or whether this small intersection should be skipped.

Best regards,
Nazli
