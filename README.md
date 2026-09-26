# Google Play Store Analytics — Internship Tasks (Streamlit)

One file, one app: `app.py` builds all 6 required visualizations from
`Play_Store_Data.csv` and `User_Reviews.csv`, each shown only inside its
required IST time window.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).
Keep `Play_Store_Data.csv` and `User_Reviews.csv` in the same folder as `app.py`.

## What's inside

| Task | Chart | IST window |
|---|---|---|
| 1 | Free vs Paid radar, top 5 categories | 1–2 PM |
| 2 | Size vs Rating density (hex-style) | 5–7 PM |
| 3 | Category→Type→Rating Band sunburst | 6–8 PM |
| 4 | Monthly installs calendar heatmap | 6–9 PM |
| 5 | Monthly/cumulative installs streamgraph | 4–6 PM |
| 6 | Category performance clustered heatmap | 3–5 PM |

Each section shows an "outside its display window" message when opened
outside its IST hours — reload the page (or interact with any widget) to
re-check the current time.

## Data limitations (also shown in-app under "Notes & data limitations")

- The dataset has **no Country column** — Task 3 uses one placeholder
  root node instead of real countries.
- The dataset has **no install-history field** — Tasks 4 & 5 build a
  "monthly installs" proxy from each app's current Installs value,
  bucketed by its Last-Updated month.

See `Internship_Report.pdf` for the full write-up of every assumption,
proxy, and filter used.
