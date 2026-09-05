/**
 * Presentational methodology data for the How It Works page.
 *
 * The ML-C/ML-F-era modules that lived here (categorization, evaluation,
 * forecasting, humanInLoop) were removed rather than left in place: they
 * transcribed numbers from a superseded selection record, and an unrendered
 * module full of stale evidence is a trap for the next person to open this
 * directory. The decisions they documented remain in the repository, in the
 * committed reports themselves (reports/ml/ML_C_*, reports/ml/ML_F_*), which
 * is where historical evidence belongs.
 *
 * Current categorization evidence lives in ./mlg.
 */
export * from "@/data/methodology/claims";
export * from "@/data/methodology/mlg";
export * from "@/data/methodology/pipeline";
