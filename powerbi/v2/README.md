# PlainCents Power BI Starter (V2)

## What this is

There is no ready-made `.pbix` (a saved Power BI report) or `.pbit` (a Power
BI report *template*) for PlainCents V2 today. This folder is a **Power BI
Starter package** instead: documentation that tells you exactly what to load
and how to lay it out, plus one real, safe, reusable asset (a color theme).
It is not a substitute for a template — it is the honest alternative when no
template exists yet.

If a real `.pbit`/`.pbix` is ever built for V2, this README stays useful as
its schema/visual-mapping documentation; nothing here needs to be thrown
away.

## Why no template exists yet

The only Power BI artifact this repository has ever carried is
[`../plaincents_theme.json`](../plaincents_theme.json) (duplicated here as
[`plaincents_theme.json`](./plaincents_theme.json) for convenience) — a
color-theme file, not a report. `viz/powerbi_export.py` (V1) only ever wrote
CSVs to `data/exports/`; it never produced or shipped a `.pbix`/`.pbit`
either. No `.pbix`/`.pbit` file has ever existed in this repository's git
history. Rather than fabricate one or claim a template exists, this package
documents the exact tables and a suggested visual layout so you can build
(or rebuild) the report yourself in a few minutes — see [SCHEMA.md](./SCHEMA.md)
and [VISUALS.md](./VISUALS.md).

## The workflow

```
PlainCents (running locally)
        |
        v
Dashboard -> Export for Power BI -> Download data pack
        |
        v
Unzip plaincents_export_YYYY-MM-DD.zip
        |
        v
Power BI Desktop -> Get Data -> Text/CSV (or Folder) -> load each table
        |
        v
Apply relationships/visuals from SCHEMA.md / VISUALS.md
(optionally apply plaincents_theme.json: View -> Themes -> Browse for themes)
        |
        v
Your PlainCents dashboard, in Power BI Desktop
```

Later, when your PlainCents data changes:

1. Click **Export for Power BI** again in the app to get a fresh ZIP.
2. Extract it over (or alongside) the old files.
3. In Power BI Desktop, click **Refresh** (Home ribbon) — your existing
   visuals update from the new files. You do not need to rebuild anything.

## What this is NOT

- **Not a live connection.** Power BI never talks to PlainCents directly.
  Every load is a snapshot of whatever PlainCents' database contained at the
  moment you clicked Download — refreshing Power BI re-reads the files on
  disk, it does not re-query the app.
- **Not automatic.** Nothing launches Power BI Desktop for you, and nothing
  generates a `.pbix` file programmatically. You do the Get Data step
  yourself, once, and then just re-point/refresh on future exports.
- **Not investment advice.** The Portfolio tables are informational, exactly
  as they are inside the app itself.

## Step by step

1. **Download the data pack.** In PlainCents, go to Dashboard → Export for
   Power BI → Download data pack. This saves a ZIP containing four CSVs:
   `transactions.csv`, `category_summary.csv`, `portfolio.csv`,
   `forecast.csv` (see [SCHEMA.md](./SCHEMA.md) for exact columns).
2. **Extract the ZIP** to a folder you'll remember (e.g.
   `Documents/PlainCents Power BI/`).
3. **Open Power BI Desktop.** (Not included with PlainCents — install it
   separately from Microsoft if you don't already have it.)
4. **Get Data → Text/CSV**, and load each of the four CSVs (or **Get Data →
   Folder**, pointed at the extracted folder, if you'd rather load all four
   at once).
5. **Set up relationships and visuals** — see [SCHEMA.md](./SCHEMA.md) for
   the tables/columns and [VISUALS.md](./VISUALS.md) for a suggested layout
   (spending trend, category breakdown, portfolio allocation, and so on).
6. **Optionally apply the PlainCents color theme**: View → Themes → Browse
   for themes → select [`plaincents_theme.json`](./plaincents_theme.json).
7. **Save your `.pbix`.** This file is now yours — Power BI Desktop's own
   Save, same as any other report.
8. **On future PlainCents updates**: download a new data pack, extract it
   over the same folder, and click Refresh in Power BI Desktop. Your
   visuals and layout stay exactly as you left them.
