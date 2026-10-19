# Manuscript layout and figure review

Reviewed on 19 October 2026.

## IEEE format

The manuscript uses IEEEtran 1.8b in journal mode, with explicit 10-point text and US Letter paper. Two columns, numbered pages, IEEE section headings, figure captions below figures, table captions above tables, and the IEEEtran bibliography style are retained. No custom page margins, compressed line spacing, or negative spacing are introduced.

The text and figure labels use TeX Gyre Termes, a Times-compatible serif family. Figure headings and panel labels use regular weight. The figure fonts are renamed TrueType conversions (Termes Figure) of the same regular and italic outlines, avoiding CFF/TrueType embedding mismatches in Matplotlib PDFs. The renderer includes the original regular and italic font files, the converted files, a reproducible conversion script, and their GUST license under `assets/fonts`, so the font does not depend on the operating system. Mathematical labels use the compatible STIX math family. The text font does not change the IEEE page geometry.

Sources checked:

- [IEEE Author Center template guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/).
- [IEEEtran maintainer](https://www.michaelshell.org/tex/ieeetran/) and [CTAN distribution](https://ctan.org/pkg/ieeetran): current class version 1.8b.
- [IEEE sample journal article](https://ieee-ims.org/files/ieeeims/media/documents/bare_jrnl_new_sample4.pdf): IEEEtran journal structure.
- [TNNLS Information for Authors, September 2026](https://doi.org/10.1109/TNNLS.2026.3723918): current issue located, but its full instructions were blocked by IEEE Xplore access verification. Therefore this review confirms the IEEE class and the implemented layout, not every current journal-specific submission requirement. Current page limits and overlength conditions should be checked in the submission portal; historical instructions are not used as current policy.

## Obsolete-content review

- Removed the unused perpendicular-bisector symbol and display, and the outdated assertion that Figure 1 distinguishes that construction from its shared face.
- Retained the definitions of closed Voronoi cells, shared boundaries, maximum modulus, and saturation boundary: these are used by the tie rule, diagnostics, and theoretical analysis.
- Retained the geometric half-plane and shared-face calculations in the rendering/checking code, where they are still needed to construct and validate exact cells.
- Replaced a hard-coded definition number with a LaTeX reference.
- Restored the current-state argument and set-membership notation in the activation-cost equation, so it no longer equates a single-valued tie-handled decision with a set-valued argmin.
- Clarified that the hosted replication checked numerical figure inputs; the latest typography was rendered locally.
- Checked all figure and table references against the five current figures and three current tables. No references to the former 2-by-2 alphabet panel or obsolete fixed-cardinality timing table remain.

The numerical datasets, table values, theorem hypotheses, and proofs are unchanged. Compilation and numerical records are in `validation/`.

Five missing BibTeX field separators were also repaired; bibliography generation is checked separately from the LaTeX layout log.
