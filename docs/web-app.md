# Web app
Live: https://rishabh-bgp.github.io/boston-house-price-prediction/
* **Inference:** `app.js` walks every tree in `model.json` (features cast with `Math.fround`, matching scikit-learn's
  float32 comparison), sums `base + lr × Σ leaves`, exponentiates.
* **Design:** classic Americana palette (navy #1f2f4a, deep red #8b1e1e, parchment #f6efe0, sunset gold #e8a33d, brown
  text), Playfair Display headings, small-caps serif subtitles, Inter body. The sunset, colonial-house roofline, tree,
  waving striped flag, ribbon banner and star dividers are original inline SVG/CSS. No company names or logos.
* **Features:** 8 main sliders + 4 more in a disclosure, three real tracts as presets, estimate in 1970 dollars,
  quantile range bar, histogram of all 506 tracts with your estimate marked, percentile, fair-housing note.
* **Accessibility:** labelled sliders with live values, `aria-live` estimate, focus outlines, reduced-motion support.
