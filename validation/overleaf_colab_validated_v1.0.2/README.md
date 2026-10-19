# Overleaf validation of the Colab-validated v1.0.2 ZIP

On 19 October 2026, `Complex_Voronoi_Overleaf_1.0.2_Colab_Validated_6f3dbfa.zip` was imported as a new Overleaf project and `main.tex` compiled successfully.

- [Validated Overleaf project](https://www.overleaf.com/project/6ad60897f64e207c3401375d) (requires appropriate Overleaf account access).
- [Exact ZIP tested](https://github.com/Fideliszan/complex-valued-voronoi-hopfield/releases/download/v1.0.2/Complex_Voronoi_Overleaf_1.0.2_Colab_Validated_6f3dbfa.zip).
- Compilation: **14 pages, 0 errors, 0 warnings**, as reported by the Overleaf log panel.
- All five paper figures are preserved, on pages 3, 5, 8, 10, and 12. All 14 PDF pages match the previously validated Overleaf PDF pixel for pixel when both are rendered with Poppler at a maximum dimension of 1200 pixels and compared in RGB. This verifies figure content and page layout at that resolution; it does not assert byte-identical PDF files.
- The imported file tree includes `reproducibility/validation/colab_v1.0.2/README.md` and `validation.json`; the Colab validation README was opened and checked in the editor.

[Structured verification record](validation.json) includes the package hash, source commit, PDF hash, reference project, and per-page comparison results.

This comparison concerns the compiled manuscript packaged in the ZIP. It is separate from the Colab-generated PNG comparisons, whose rendering differences and timing limitations remain documented in the [Colab record](../colab_v1.0.2/README.md). The existing release ZIP is unchanged; this subsequent Overleaf validation record is published in the repository.
